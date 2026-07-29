import os
import json
import torch
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from audiomentations import (
    Compose, SomeOf, AddGaussianSNR, TimeStretch, PitchShift, AddBackgroundNoise,
    AddShortNoises, ApplyImpulseResponse
)
from audiomentations.core.audio_loading_utils import load_sound_file
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf

# 基本設定

sr = 16000

# --- 第一層增強 (nlpaug: 頻譜層級，如VTLP) ---
augment_stage1 = naf.Sometimes([
    naa.VtlpAug(sampling_rate=sr, zone=(0.0, 1.0), coverage=1.0, factor=(0.9, 1.1)),
], aug_p=0.4)

# --- 第二層增強 (audiomentations: 波形層級) ---
augment_stage2 = Compose([
    AddGaussianSNR(min_snr_in_db=10, max_snr_in_db=30, p=0.2),
    TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
    AddBackgroundNoise(
        sounds_path="/mnt/usr1/ethan1017/MLHW3/background_noises",
        min_snr_in_db=10,
        max_snr_in_db=30.0,
        p=0.4
    ),
    AddShortNoises(
        sounds_path="/mnt/usr1/ethan1017/MLHW3/short_noises",
        min_snr_in_db=10,
        max_snr_in_db=30.0,
        noise_rms="relative_to_whole_input",
        min_time_between_sounds=2.0,
        max_time_between_sounds=8.0,
        p=0.3
    ),
    ApplyImpulseResponse(ir_path="/mnt/usr1/ethan1017/MLHW3/rir", p=0.4)
])

# 載入資料集與模型

dataset = load_dataset("json", data_files={"train": "/mnt/usr1/ethan1017/MLHW3/train2.jsonl"}, split="train")

model_name = "openai/whisper-large-v3-turbo"
processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 音訊預處理 + 資料增強

def prepare_example(example):
    # 讀取音檔
    speech_array, sr_orig = torchaudio.load(example["audio_filepath"])
    if sr_orig != sr:
        resampler = torchaudio.transforms.Resample(sr_orig, sr)
        speech_array = resampler(speech_array)

    speech_array = speech_array.squeeze().numpy()

    # ======= Stage 1: 頻譜層增強 (VTLP) =======
    try:
        speech_array = augment_stage1.augment(speech_array)
        if isinstance(speech_array, list):
            speech_array = speech_array[0]
    except Exception as e:
        print(f"[WARN] Stage1 VTLP augment failed for {example['audio_filepath']}: {e}")

    # ======= Stage 2: 波形層增強 =======
    try:
        speech_array = augment_stage2(samples=speech_array, sample_rate=sr)
    except Exception as e:
        print(f"[WARN] Stage2 augment failed for {example['audio_filepath']}: {e}")

    # ======= Whisper 特徵轉換 =======
    input_features = processor.feature_extractor(speech_array, sampling_rate=sr).input_features[0]
    example["input_features"] = input_features

    # ======= 標籤 =======
    labels = processor.tokenizer(example["text"], max_length=448, truncation=True).input_ids
    example["labels"] = labels

    return example

# 套用映射與 DataCollator

dataset = dataset.map(prepare_example)

def data_collator(batch):
    input_features = [torch.tensor(item["input_features"], dtype=torch.float) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    input_features = pad_sequence(input_features, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_features": input_features, "labels": labels}

# 訓練設定

training_args = TrainingArguments(
    output_dir="/mnt/usr1/ethan1017/MLHW3/baseline",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=12,
    num_train_epochs=6,
    learning_rate=1e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=1500,
    fp16=True,
    evaluation_strategy="no",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

if __name__ == "__main__":
    trainer.train()
