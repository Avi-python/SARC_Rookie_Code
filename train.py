#!/usr/bin/env python3
"""
Whisper Multi-GPU Fine-Tuning Script
====================================
Features:
- Reads JSONL audio dataset (train.jsonl / dev.jsonl) and loads audio files from `audio_path`.
- Uses `clean_text` without language filtering.
- Unified `prepare_batch` function incorporating audio loading & SpecAugment (Frequency & Time Masking).
- Full multi-GPU training support via PyTorch DDP / `torchrun` / Hugging Face Trainer.

Usage:
  # 1. Multi-GPU training using torchrun (recommended for multi-GPU):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py

  # 2. Single-GPU execution:
  python train.py
"""

import os
import sys
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
from tokenizers import processors
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
    parser.add_argument("--language", type=str, default="<|formosan|>", help="Target language token or code (default: <|formosan|>)")
    parser.add_argument("--init_lang_from", type=str, default="<|id|>", help="Existing language token used to warm-start custom language token embedding (default: <|id|>)")
    parser.add_argument("--task", type=str, default="transcribe", help="Task (transcribe or translate)")
    parser.add_argument("--data_dir", type=str, default="/nfs/DGX2/Workspace/avi/FormosanBank/asr_dataset/", help="JSONL dataset root directory containing train.jsonl and dev.jsonl")
    parser.add_argument("--train_file", type=str, default=None, help="Path to train.jsonl (default: <data_dir>/train.jsonl)")
    parser.add_argument("--dev_file", type=str, default=None, help="Path to dev.jsonl (default: <data_dir>/dev.jsonl)")
    parser.add_argument("--output_dir", type=str, default="/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-v2", help="Output directory")
    parser.add_argument("--epochs", type=float, default=6, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform (overrides epochs)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=300, help="Evaluation steps (should match or divide save_steps)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per GPU for evaluation")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience (number of evaluations without improvement before stopping, 0 to disable)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0, help="Minimum change in monitored metric to qualify as an improvement")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional maximum number of samples from train dataset (useful for smoke testing)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional maximum number of samples from dev dataset to evaluate")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=True, help="Resume from last checkpoint if exists")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of CPU workers for on-the-fly DataLoader preprocessing")
    parsed = parser.parse_known_args()[0]

    if parsed.train_file is None:
        parsed.train_file = os.path.join(parsed.data_dir, "train.jsonl")
    if parsed.dev_file is None:
        parsed.dev_file = os.path.join(parsed.data_dir, "dev.jsonl")
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

# %% 載入 Whisper Processor 與 Model 並註冊自訂語言 Token (如 <|formosan|>)
lang_token = args.language if args.language.startswith("<|") else f"<|{args.language}|>"
logging.info(f"Loading Whisper processor ({args.model_name})...")
processor = WhisperProcessor.from_pretrained(args.model_name)

logging.info(f"Loading Whisper model ({args.model_name})...")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float32)

# 新增自訂語言 token 至 tokenizer 並擴充模型 embedding
num_added = processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": [lang_token]}
)
lang_token_id = processor.tokenizer.convert_tokens_to_ids(lang_token)

if num_added > 0:
    logging.info(f"Added new language token '{lang_token}' (id={lang_token_id}). Resizing model token embeddings to {len(processor.tokenizer)}...")
    model.resize_token_embeddings(len(processor.tokenizer))
    init_token_id = processor.tokenizer.convert_tokens_to_ids(args.init_lang_from)
    if init_token_id != processor.tokenizer.unk_token_id:
        with torch.no_grad():
            model.model.decoder.embed_tokens.weight[lang_token_id] = (
                model.model.decoder.embed_tokens.weight[init_token_id].clone()
            )
        logging.info(f"Initialized '{lang_token}' embedding from '{args.init_lang_from}' (id={init_token_id}).")
else:
    logging.info(f"Language token '{lang_token}' (id={lang_token_id}) already exists in tokenizer.")

# 設定 tokenizer post-processor 自動加上 prefix:
# <|startoftranscript|> <|formosan|> <|transcribe|> <|notimestamps|> ... <|endoftext|>
sot_token = "<|startoftranscript|>"
task_token = f"<|{args.task}|>"
notimestamps_token = "<|notimestamps|>"
eos_token = processor.tokenizer.eos_token

prefix_tokens = [sot_token, lang_token, task_token, notimestamps_token]
prefix_ids = processor.tokenizer.convert_tokens_to_ids(prefix_tokens)
eos_id = processor.tokenizer.eos_token_id

prefix_template = " ".join([f"{tok}:0" for tok in prefix_tokens])
processor.tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
    single=f"{prefix_template} $A:0 {eos_token}:0",
    pair=f"{prefix_template} $A:0 $B:1 {eos_token}:1",
    special_tokens=[(eos_token, eos_id), *zip(prefix_tokens, prefix_ids)],
)

# 將自訂語言 token 註冊進 generation_config.lang_to_id
if hasattr(model.generation_config, "lang_to_id") and model.generation_config.lang_to_id is not None:
    model.generation_config.lang_to_id[lang_token] = lang_token_id
model.config.forced_decoder_ids = None
model.generation_config.language = lang_token
model.generation_config.task = args.task
model.generation_config.forced_decoder_ids = None
logging.info(f"Model and Processor configured successfully with target language token: {lang_token} (id={lang_token_id})!")


# %% SpecAugment 轉換設定 (頻譜遮罩增強: Frequency & Time Masking)
freq_masking = T.FrequencyMasking(freq_mask_param=27)
time_masking = T.TimeMasking(time_mask_param=100)


def prepare_batch(batch, is_train=True):
    """
    音訊即時預處理 + 讀取 audio_path + 動態 SpecAugment (僅訓練集) + 標籤 Tokenizer (clean_text)
    使用 set_transform 在 DataLoader 多核心 CPU 即時平行運算，不佔用任何硬碟快取。
    """
    input_features_list = []
    labels_list = []

    audio_path_list = batch["audio_path"] if isinstance(batch.get("audio_path"), list) else [batch.get("audio_path")]
    text_list = batch["clean_text"] if isinstance(batch.get("clean_text"), list) else [batch.get("clean_text")]

    for audio_path, text in zip(audio_path_list, text_list):
        speech_array = None
        if audio_path is not None and os.path.exists(audio_path):
            try:
                # 1. 由 audio_path 讀取音訊檔
                speech_array, sr_orig = torchaudio.load(audio_path)
                # 若為多聲道則轉為單聲道
                if speech_array.shape[0] > 1:
                    speech_array = speech_array.mean(dim=0, keepdim=True)
                # 2. 若取樣率非 16000Hz 則重取樣
                if sr_orig != sr:
                    resampler = T.Resample(sr_orig, sr)
                    speech_array = resampler(speech_array)
                speech_array = speech_array.squeeze(0).numpy()
            except Exception:
                speech_array = None

        if speech_array is None or len(speech_array) == 0 or not text:
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


# %% 載入 JSONL 資料集 (Train & Dev)
if not os.path.exists(args.train_file):
    raise FileNotFoundError(f"Train JSONL file not found: {args.train_file}")

logging.info(f"Loading train dataset from {args.train_file} via Hugging Face `load_dataset`...")
train_dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")
if args.max_train_samples is not None and args.max_train_samples < len(train_dataset):
    logging.info(f"Subsampling train dataset to {args.max_train_samples} records.")
    train_dataset = train_dataset.select(range(args.max_train_samples))
logging.info(f"Train dataset loaded successfully! Total records: {len(train_dataset)}")

eval_dataset = None
if args.dev_file and os.path.exists(args.dev_file):
    logging.info(f"Loading dev dataset from {args.dev_file} via Hugging Face `load_dataset`...")
    eval_dataset = load_dataset("json", data_files={"dev": args.dev_file}, split="dev")
    if args.max_eval_samples is not None and args.max_eval_samples < len(eval_dataset):
        logging.info(f"Subsampling dev dataset to {args.max_eval_samples} records for evaluation.")
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    logging.info(f"Dev dataset loaded successfully! Total records: {len(eval_dataset)}")
else:
    logging.warning(f"No dev JSONL file found at {args.dev_file}. Evaluation and early stopping will be disabled.")


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
    max_steps=args.max_steps,
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
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    logging.info("Training finished! Saving final model and training state...")
    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()  # Saves full trainer_state.json directly to args.output_dir
    logging.info(f"Model and processor saved to '{final_model_dir}', trainer_state saved to '{args.output_dir}'!")
