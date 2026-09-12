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
import inspect
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
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
    default_base_model = (
        "/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-v2/final_model-evaluation/final_model"
        if os.path.exists("/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-v2/final_model-evaluation/final_model")
        else "openai/whisper-large-v3-turbo"
    )
    parser.add_argument("--model_name", type=str, default=default_base_model, help="Base model name or path")
    parser.add_argument("--language", type=str, default="id", help="Target language code (default: id for Indonesian / 印尼語)")
    parser.add_argument("--task", type=str, default="transcribe", help="Task (transcribe or translate)")
    parser.add_argument("--language_group", type=str, default="dru", help="Target language ID group (e.g. 'dru', or comma-separated 'dru,tsu'). Set to 'all' for all languages.")
    parser.add_argument("--data_dir", type=str, default="/nfs/DGX2/Workspace/avi/processed_data_independent3/", help="Parquet dataset root directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: whisper-finetuned-<language_group>)")
    parser.add_argument("--epochs", type=float, default=6, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=300, help="Evaluation steps (should match or divide save_steps)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per GPU for evaluation")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience (number of evaluations without improvement before stopping, 0 to disable)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0, help="Minimum change in monitored metric to qualify as an improvement")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional maximum number of samples from dev dataset to evaluate")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=True, help="Resume from last checkpoint if exists")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of CPU workers for on-the-fly DataLoader preprocessing")
    parsed = parser.parse_known_args()[0]

    if parsed.output_dir is None:
        if parsed.language_group and parsed.language_group.lower() != "all":
            group_slug = parsed.language_group.replace(",", "_").replace(" ", "_")
            parsed.output_dir = f"/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-{group_slug}"
        else:
            parsed.output_dir = "/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-v2"
    return parsed


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


def prepare_batch(batch, is_train=True):
    """
    音訊即時預處理 + Parquet 解析 + 動態 SpecAugment (僅訓練集) + 標籤 Tokenizer (印尼語)
    使用 set_transform 在 DataLoader 多核心 CPU 即時平行運算，不佔用任何硬碟快取。
    """
    input_features_list = []
    labels_list = []

    audio_bytes_list = batch["audio_bytes"] if isinstance(batch.get("audio_bytes"), list) else [batch.get("audio_bytes")]
    text_list = batch["text"] if isinstance(batch.get("text"), list) else [batch.get("text")]

    for audio_raw, text in zip(audio_bytes_list, text_list):
        speech_array = None
        if audio_raw is not None:
            try:
                # 1. 提取 Parquet raw bytes 並由記憶體 buffer 讀取
                audio_bytes = np.array(audio_raw, dtype=np.int8).view(np.uint8).tobytes()
                speech_array, sr_orig = torchaudio.load(io.BytesIO(audio_bytes))
                # 2. 若取樣率非 16000Hz 則重取樣
                if sr_orig != sr:
                    resampler = T.Resample(sr_orig, sr)
                    speech_array = resampler(speech_array)
                speech_array = speech_array.squeeze().numpy()
            except Exception:
                speech_array = None

        if speech_array is None or len(speech_array) == 0:
            # 若音訊損壞或讀取失敗，使用 1 秒靜音作為 fallback，標籤設為 -100 (計算 loss 時自動忽略)
            speech_array = np.zeros(sr, dtype=np.float32)
            input_features = processor.feature_extractor(speech_array, sampling_rate=sr).input_features[0]
            label_ids = [-100]
        else:
            # 3. Whisper 特徵轉換 (Log-Mel Spectrogram)
            input_features = processor.feature_extractor(speech_array, sampling_rate=sr).input_features[0]

            # 4. 動態 SpecAugment (僅在訓練集套用，每次 epoch 生成全新隨機頻譜遮罩，防止過擬合；驗證集使用原始特徵)
            if is_train:
                try:
                    features_tensor = torch.tensor(input_features)
                    features_tensor = freq_masking(features_tensor)
                    features_tensor = freq_masking(features_tensor)
                    features_tensor = time_masking(features_tensor)
                    features_tensor = time_masking(features_tensor)
                    input_features = features_tensor.numpy()
                except Exception:
                    pass

            # 5. 標籤 Tokenizer
            label_ids = processor.tokenizer(text, max_length=448, truncation=True).input_ids

        input_features_list.append(input_features)
        labels_list.append(label_ids)

    return {"input_features": input_features_list, "labels": labels_list}


def prepare_train_batch(batch):
    return prepare_batch(batch, is_train=True)


def prepare_eval_batch(batch):
    return prepare_batch(batch, is_train=False)


def filter_parquet_files(files, language_group, data_dir):
    if not language_group or language_group.lower() == "all":
        return sorted(files)
    groups = [g.strip() for g in language_group.split(",") if g.strip()]
    matched = []
    for f in files:
        rel = os.path.relpath(f, data_dir)
        top_folder = rel.split(os.sep)[0]
        if any(top_folder == g or top_folder.startswith(f"{g}_") or top_folder.startswith(g) for g in groups):
            matched.append(f)
    return sorted(matched)


# %% 搜尋並載入 Parquet 資料集 (Train & Dev)
logging.info(f"Searching train parquet files under {args.data_dir} (language_group: '{args.language_group}')...")
all_train_parquet_files = glob.glob(f"{args.data_dir}/**/split=train/**/*.parquet", recursive=True)
train_parquet_files = filter_parquet_files(all_train_parquet_files, args.language_group, args.data_dir)
logging.info(f"Found {len(train_parquet_files)} matching train parquet files (out of {len(all_train_parquet_files)} total):")
for f in train_parquet_files:
    rel_dialect = os.path.relpath(f, args.data_dir).split(os.sep)[0]
    logging.info(f"  - Train dialect [{rel_dialect}]: {f}")

if not train_parquet_files:
    raise FileNotFoundError(f"No train parquet files found matching language_group '{args.language_group}' under directory: {args.data_dir}")

logging.info("Loading train dataset via Hugging Face `load_dataset`...")
train_dataset = load_dataset("parquet", data_files={"train": train_parquet_files}, split="train")
logging.info(f"Train dataset loaded successfully! Total records: {len(train_dataset)}")

logging.info(f"Searching dev parquet files under {args.data_dir} (language_group: '{args.language_group}')...")
all_dev_parquet_files = glob.glob(f"{args.data_dir}/**/split=dev/**/*.parquet", recursive=True)
dev_parquet_files = filter_parquet_files(all_dev_parquet_files, args.language_group, args.data_dir)
logging.info(f"Found {len(dev_parquet_files)} matching dev parquet files (out of {len(all_dev_parquet_files)} total):")
for f in dev_parquet_files:
    rel_dialect = os.path.relpath(f, args.data_dir).split(os.sep)[0]
    logging.info(f"  - Dev dialect [{rel_dialect}]: {f}")

eval_dataset = None
if dev_parquet_files:
    logging.info("Loading dev dataset via Hugging Face `load_dataset`...")
    eval_dataset = load_dataset("parquet", data_files={"dev": dev_parquet_files}, split="dev")
    if args.max_eval_samples is not None and args.max_eval_samples < len(eval_dataset):
        logging.info(f"Subsampling dev dataset to {args.max_eval_samples} records for evaluation.")
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    logging.info(f"Dev dataset loaded successfully! Total records: {len(eval_dataset)}")
else:
    logging.warning(f"No dev parquet files found under {args.data_dir} matching '{args.language_group}' with split=dev. Evaluation and early stopping will be disabled.")


# %% 訓練與評估設定 (支援多 GPU 分散式訓練 DDP + 多 CPU 核心 DataLoader 即時資料處理)
eval_args = {}
callbacks = []

if eval_dataset is not None:
    # 相容不同版本的 transformers (4.41+ 使用 eval_strategy，舊版使用 evaluation_strategy)
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        eval_args["eval_strategy"] = "steps"
    else:
        eval_args["evaluation_strategy"] = "steps"
    eval_args["eval_steps"] = args.eval_steps
    eval_args["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    eval_args["load_best_model_at_end"] = True
    eval_args["metric_for_best_model"] = "eval_loss"
    eval_args["greater_is_better"] = False

    if args.early_stopping_patience is not None and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
        logging.info(f"EarlyStoppingCallback enabled with patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}")

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
    dataloader_num_workers=args.num_proc,
    dataloader_prefetch_factor=2 if args.num_proc > 0 else None,
    remove_unused_columns=False,
    report_to="none",
    logging_first_step=True,
    **eval_args,
)


# %% 套用即時轉換 (使用 set_transform 免除硬碟快取，由 DataLoader 多核心 CPU 即時平行處理)
logging.info(f"Applying dynamic on-the-fly preprocessing to train dataset (num_workers={args.num_proc})...")
train_dataset.set_transform(prepare_train_batch)

if eval_dataset is not None:
    logging.info("Applying on-the-fly preprocessing to dev dataset (clean audio, no SpecAugment)...")
    eval_dataset.set_transform(prepare_eval_batch)

logging.info("Dataset set_transform configured successfully! No disk cache will be generated.")


def data_collator(batch):
    input_features = [torch.tensor(item["input_features"], dtype=torch.float32) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    input_features = torch.stack(input_features)  # shape: (batch, 128, 3000)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # variable length padding
    return {"input_features": input_features, "labels": labels}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=processor.tokenizer,
    callbacks=callbacks,
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
