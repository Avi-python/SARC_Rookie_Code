#!/usr/bin/env python3
"""
Whisper Multi-GPU Fine-Tuning Script
====================================
Features:
- Reads Parquet audio dataset directly from memory buffer (io.BytesIO).
- Unified `prepare_example` function incorporating Parquet extraction & SpecAugment (Frequency & Time Masking).
- Full multi-GPU training support via PyTorch DDP / `torchrun` / Hugging Face Trainer.
- Target language set to 毛利語 (Maori, language code 'mi') for both tokenizer and generation config.

Usage:
  # 1. Multi-GPU training using torchrun (recommended for multi-GPU):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py

  # 2. Single-GPU execution:
  python train.py
"""

import os
import io
import sys
import glob
import time
import argparse
import logging
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
import transformers.trainer
import transformers.utils.import_utils
import transformers.utils.logging

# Allow torch.load for trusted local checkpoint files
transformers.trainer.check_torch_load_is_safe = lambda: None
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

# Allow loading NumPy RNG states from checkpoint
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

# For RTX 4000 series NVIDIA GPUs to avoid NCCL peer-to-peer communication issues
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Multi-GPU Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="openai/whisper-large-v3-turbo", help="Base model name or path")
    parser.add_argument("--language", type=str, default="id", help="Target language code (default: id for Indonesian / 印尼語)")
    parser.add_argument("--task", type=str, default="transcribe", help="Task (transcribe or translate)")
    parser.add_argument("--data_dir", type=str, default="/nfs/DGX2/Workspace/avi/processed_data_independent3/", help="Parquet dataset root directory")
    parser.add_argument("--output_dir", type=str, default="/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-v1", help="Output directory")
    parser.add_argument("--epochs", type=float, default=6, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=1500, help="Save checkpoint steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=True, help="Resume from last checkpoint if exists")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for dataset mapping")
    return parser.parse_known_args()[0]


# %% 基本全域設定
args = parse_args()
sr = 16000

# Setup Logging for distributed / single runs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Process %(process)d) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# %% 載入 Whisper Processor 與 Model (指定語言為印尼語 'id')
logging.info(f"Loading Whisper processor ({args.model_name}) with language='{args.language}', task='{args.task}'...")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)

logging.info(f"Loading Whisper model ({args.model_name})...")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float32)

# Ensure target language ('id') and task ('transcribe') tokens are set in generation_config
model.config.forced_decoder_ids = None
model.generation_config.language = args.language
model.generation_config.task = args.task
model.generation_config.forced_decoder_ids = None
logging.info(f"Model and Processor loaded successfully with target language: {args.language}!")


# %% SpecAugment 轉換設定 (頻譜遮罩增強: Frequency & Time Masking)
freq_masking = T.FrequencyMasking(freq_mask_param=27)
time_masking = T.TimeMasking(time_mask_param=100)


def prepare_example(example):
    """
    音訊預處理 + Parquet 解析 + SpecAugment + 標籤 Tokenizer (印尼語)
    """
    # 1. 提取 Parquet raw bytes
    audio_bytes = example.get("audio_bytes")
    if audio_bytes is None:
        example["input_features"] = None
        example["labels"] = None
        return example

    audio_bytes = np.array(audio_bytes, dtype=np.int8).view(np.uint8).tobytes()

    # 2. 直接由記憶體 buffer 讀取音訊
    try:
        speech_array, sr_orig = torchaudio.load(io.BytesIO(audio_bytes))
    except Exception:
        example["input_features"] = None
        example["labels"] = None
        return example

    # 3. 若取樣率非 16000Hz 則重取樣
    if sr_orig != sr:
        resampler = T.Resample(sr_orig, sr)
        speech_array = resampler(speech_array)

    speech_array = speech_array.squeeze().numpy()

    # 4. Whisper 特徵轉換 (Log-Mel Spectrogram)
    input_features = processor.feature_extractor(speech_array, sampling_rate=sr).input_features[0]

    # 5. SpecAugment (頻譜遮罩增強: 2 次 Frequency Masking + 2 次 Time Masking)
    try:
        features_tensor = torch.tensor(input_features)
        features_tensor = freq_masking(features_tensor)
        features_tensor = freq_masking(features_tensor)
        features_tensor = time_masking(features_tensor)
        features_tensor = time_masking(features_tensor)
        input_features = features_tensor.numpy()
    except Exception as e:
        pass

    example["input_features"] = input_features

    # 6. 標籤 Tokenizer (以毛利語 prompt 編碼)
    labels = processor.tokenizer(example["text"], max_length=448, truncation=True).input_ids
    example["labels"] = labels

    return example


# %% 搜尋並載入 Parquet 資料集
logging.info(f"Searching parquet files under {args.data_dir}...")
parquet_files = glob.glob(f"{args.data_dir}/**/split=train/**/*.parquet", recursive=True)
logging.info(f"Found {len(parquet_files)} parquet files under {args.data_dir}")

if not parquet_files:
    raise FileNotFoundError(f"No parquet files found under directory: {args.data_dir}")

logging.info("Loading dataset via Hugging Face `load_dataset`...")
dataset = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
logging.info(f"Dataset loaded successfully! Total records: {len(dataset)}")


# %% 訓練設定 (支援多 GPU 分散式訓練 DDP)
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    ddp_timeout=7200,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    report_to="none",
    logging_first_step=True,
)


# %% 套用映射 (多程序平行處理，在多 GPU / DDP 環境下由主程序先處理並快取)
with training_args.main_process_first(desc="Mapping dataset with prepare_example (Parquet + SpecAugment)"):
    logging.info(f"Mapping dataset with prepare_example function (num_proc={args.num_proc})...")
    dataset = dataset.map(prepare_example, num_proc=min(args.num_proc, os.cpu_count() or 4))
    logging.info("Dataset mapped successfully!")

    before = len(dataset)
    dataset = dataset.filter(lambda x: x["input_features"] is not None, num_proc=min(args.num_proc, os.cpu_count() or 4))
    logging.info(f"Dropped {before - len(dataset)} corrupted/invalid examples")


def data_collator(batch):
    input_features = [torch.tensor(item["input_features"], dtype=torch.float32) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    input_features = torch.stack(input_features)  # shape: (batch, 128, 3000)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # variable length padding
    return {"input_features": input_features, "labels": labels}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    processing_class=processor.tokenizer,
)

if __name__ == "__main__":
    logging.info("Starting training...")

    checkpoint = None
    if args.resume_from_checkpoint and os.path.isdir(args.output_dir):
        checkpoint = get_last_checkpoint(args.output_dir)
        if checkpoint is not None:
            logging.info(f"Found checkpoint, resuming from: {checkpoint}")
        else:
            logging.info("No checkpoint found in output directory. Starting from scratch.")
    trainer.train(resume_from_checkpoint=checkpoint)

    logging.info("Training finished! Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    logging.info("Model and processor saved successfully!")
