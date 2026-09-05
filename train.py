# %% 載入套件
import os
import io
import json
import torch
import torchaudio
import numpy as np
import glob
import accelerate
import time
import logging
import transformers.trainer
import transformers.utils.import_utils
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from audiomentations import (
    Compose, SomeOf, AddGaussianSNR, TimeStretch, PitchShift, AddBackgroundNoise,
    AddShortNoises, ApplyImpulseResponse
)
from audiomentations.core.audio_loading_utils import load_sound_file
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf

# Allow torch.load for trusted local checkpoint files
transformers.trainer.check_torch_load_is_safe = lambda: None
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

# Allow loading NumPy RNG states from checkpoint
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

# for 4000 series nvidia GPUs
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# %% 設定 Logging (適用於背景 / nohup 執行)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# %% 基本設定
sr = 16000

# # --- 第一層增強 (nlpaug: 頻譜層級，如VTLP) ---
# augment_stage1 = naf.Sometimes([
#     naa.VtlpAug(sampling_rate=sr, zone=(0.0, 1.0), coverage=1.0, factor=(0.9, 1.1)),
# ], aug_p=0.4)

# # --- 第二層增強 (audiomentations: 波形層級) ---
# augment_stage2 = Compose([
#     AddGaussianSNR(min_snr_in_db=10, max_snr_in_db=30, p=0.2),
#     TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
#     PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
#     AddBackgroundNoise(
#         sounds_path="/mnt/usr1/ethan1017/MLHW3/background_noises",
#         min_snr_in_db=10,
#         max_snr_in_db=30.0,
#         p=0.4
#     ),
#     AddShortNoises(
#         sounds_path="/mnt/usr1/ethan1017/MLHW3/short_noises",
#         min_snr_in_db=10,
#         max_snr_in_db=30.0,
#         noise_rms="relative_to_whole_input",
#         min_time_between_sounds=2.0,
#         max_time_between_sounds=8.0,
#         p=0.3
#     ),
#     ApplyImpulseResponse(ir_path="/mnt/usr1/ethan1017/MLHW3/rir", p=0.4)
# ])

# %% 載入資料
logging.info("=== Task Started ===")
start_time = time.time()

root_dir = "/nfs/DGX2/Workspace/avi/processed_data_independent3/"
logging.info(f"Searching parquet files under {root_dir}...")
parquet_files = glob.glob(f"{root_dir}/**/split=train/**/*.parquet", recursive=True)
logging.info(f"Found {len(parquet_files)} parquet files under {root_dir}")

# %% 載入模型和資料集
if not parquet_files:
    logging.warning(f"No parquet files found under directory: {root_dir}")
else:
    logging.info("Loading dataset via Hugging Face `load_dataset`...")
    dataset = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
    logging.info(f"Dataset loaded successfully! Total records: {len(dataset)}")
    logging.info(f"Dataset columns: {dataset.column_names}")

model_name = "openai/whisper-large-v3-turbo"
logging.info(f"Loading Whisper processor ({model_name})...")
processor = WhisperProcessor.from_pretrained(model_name, language=None, task="transcribe")

logging.info(f"Loading Whisper model ({model_name})...")
model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=torch.float32)
logging.info("Model loaded successfully!")

elapsed_time = time.time() - start_time
logging.info(f"=== Task Completed in {elapsed_time:.2f} seconds ===")

# %% 音訊預處理 + 資料增強

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

# %% 音訊預處理 + 資料增強 (Parquet 版本)

def prepare_example_parquet(example):

    # 1. Extract raw bytes
    audio_bytes = example["audio_bytes"]
    audio_bytes = np.array(audio_bytes, dtype=np.int8).view(np.uint8).tobytes()
    
    # 2. Read audio directly from memory buffer
    try:
        speech_array, sr_orig = torchaudio.load(io.BytesIO(audio_bytes))
    except RuntimeError:
        # mark as invalid, handle after map()
        example["input_features"] = None
        example["labels"] = None
        return example
    
    # 3. Resample if necessary (Whisper requires 16000 Hz)
    if sr_orig != sr:
        resampler = torchaudio.transforms.Resample(sr_orig, sr)
        speech_array = resampler(speech_array)

    speech_array = speech_array.squeeze().numpy()

    # ======= Whisper 特徵轉換 =======
    input_features = processor.feature_extractor(speech_array, sampling_rate=sr).input_features[0]
    example["input_features"] = input_features

    # ======= 標籤 =======
    labels = processor.tokenizer(example["text"], max_length=448, truncation=True).input_ids
    example["labels"] = labels

    return example

# %% 套用映射與 DataCollator

logging.info(f"Mapping dataset with prepare_example_parquet function...")
dataset = dataset.map(prepare_example_parquet)
logging.info("Dataset mapped successfully!")

before = len(dataset)
dataset = dataset.filter(lambda x: x["input_features"] is not None)
logging.info(f"Dropped {before - len(dataset)} corrupted examples")

def data_collator(batch):
    input_features = [torch.tensor(item["input_features"], dtype=torch.float32) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    input_features = torch.stack(input_features)  # shape: (batch, 128, 3000) — fixed size, just stack
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # variable length — pad
    return {"input_features": input_features, "labels": labels}

# %% 訓練設定

training_args = TrainingArguments(
    output_dir="/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-v1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=12,
    num_train_epochs=6,
    learning_rate=1e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=1500,
    save_total_limit=2,
    bf16=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    processing_class=processor.tokenizer,
)

if __name__ == "__main__":
    trainer.train(resume_from_checkpoint=True)

# %%
