import os
import sys
import torch
import numpy as np
import argparse
import torchaudio
import gc
import psutil
import threading
import time
import json
import subprocess
from datetime import datetime
from collections import OrderedDict
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

# --- 新增音訊增強套件引入 ---
try:
    from audiomentations import Compose, AddGaussianSNR, TimeStretch, PitchShift, AddBackgroundNoise, AddShortNoises, ApplyImpulseResponse
    import nlpaug.augmenter.audio as naa
    import nlpaug.flow as naf
except ImportError:
    print("請先安裝增強套件: pip install audiomentations nlpaug librosa")
    sys.exit(1)

# --------- 1. 系統資源監控與環境設定 ---------
def setup_environment():
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,8,9"
    print("=== 訓練開始前系統狀況 ===")
    mem = psutil.virtual_memory()
    print(f"記憶體使用情況: 總計 {mem.total/1024**3:.1f}GB, 已使用 {mem.used/1024**3:.1f}GB")
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv"]).decode()
        print(f"GPU 狀況:\n{gpu_info}")
    except:
        print("無法取得 GPU 狀況")
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# --------- 2. 參數解析 ---------
def parse_args():
    parser = argparse.ArgumentParser(description="Taiwanese Whisper Finetuning with Augmentation")
    parser.add_argument("--train_json", type=str, default="/mnt/usr1/ethan1017/Taiwanese_ASR/data/12000hr_0726train.jsonl")
    parser.add_argument("--eval_json", type=str, default="/mnt/usr1/ethan1017/Taiwanese_ASR/data/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="/mnt/usr1/ethan1017/Taiwanese_ASR/output")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_audio_len", type=float, default=29.5)
    parser.add_argument("--min_audio_len", type=float, default=0.5)
    parser.add_argument("--cache_size_gb", type=float, default=50.0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()

# --------- 3. 快取系統 (保留原功能) ---------
class LightweightAudioCache:
    def __init__(self, max_size_gb=2.0):
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache = OrderedDict()
        self.current_size = 0
        self.lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0
    def _estimate_size(self, audio_array):
        return audio_array.nbytes if hasattr(audio_array, 'nbytes') else len(audio_array) * 4
    def _evict_lru(self, needed_size):
        while (self.current_size + needed_size > self.max_size_bytes or len(self.cache) > 500) and self.cache:
            oldest_key, oldest_data = self.cache.popitem(last=False)
            self.current_size -= self._estimate_size(oldest_data)
            del oldest_data
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.copy()
            self.miss_count += 1
            return None
    def put(self, key, value):
        with self.lock:
            if key in self.cache: return
            size = self._estimate_size(value)
            if size > self.max_size_bytes * 0.1: return
            self._evict_lru(size)
            self.cache[key] = value.copy()
            self.current_size += size
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            gc.collect()
    def get_cache_info(self):
        with self.lock:
            total = self.hit_count + self.miss_count
            return {'size': len(self.cache), 'memory_mb': self.current_size / (1024*1024), 'hit_rate': self.hit_count / total if total > 0 else 0}

audio_cache = None

def load_audio_fast(audio_path):
    global audio_cache
    if audio_cache is not None:
        cached = audio_cache.get(audio_path)
        if cached is not None: return cached
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            if not hasattr(load_audio_fast, '_resamplers'): load_audio_fast._resamplers = {}
            if sample_rate not in load_audio_fast._resamplers:
                load_audio_fast._resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = load_audio_fast._resamplers[sample_rate](waveform)
        audio_array = waveform.squeeze().numpy().astype(np.float32)
        if audio_cache is not None: audio_cache.put(audio_path, audio_array)
        return audio_array
    except: return None

def prepare_example_minimal(example):
    return {"audio_path": example["audio"]["path"], "sentence": example["sentence"], "duration": example["duration"]}

# --------- 4. 記憶體優化且含增強的 Data Collator ---------
class AugmentedDataCollator:
    def __init__(self, processor, augment=False):
        self.processor = processor
        self.augment = augment
        
        if self.augment:
            # 第一層增強 (Spectrum Level)
            self.aug_stage1 = naf.Sometimes([
                naa.VtlpAug(sampling_rate=16000, zone=(0.0, 1.0), coverage=1.0, factor=(0.9, 1.1)),
            ], aug_p=0.4)
            
            # 第二層增強 (Waveform Level)
            self.aug_stage2 = Compose([
                AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.2),
                TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
                AddBackgroundNoise(
                    sounds_path="/mnt/usr1/ethan1017/Taiwanese_ASR/background_noises",
                    min_snr_db=10, max_snr_db=30.0, p=0.4
                ),
                AddShortNoises(
                    sounds_path="/mnt/usr1/ethan1017/Taiwanese_ASR/short_noises",
                    min_snr_db=10, max_snr_db=30.0, noise_rms="relative_to_whole_input",
                    min_time_between_sounds=2.0, max_time_between_sounds=8.0, p=0.3
                ),
                ApplyImpulseResponse(ir_path="/mnt/usr1/ethan1017/Taiwanese_ASR/rir", p=0.4)
            ])

    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None
        
        input_features, labels = [], []
        for item in batch:
            speech_array = load_audio_fast(item["audio_path"])
            if speech_array is None: continue
            
            # 執行音訊增強 (僅在訓練時)
            if self.augment:
                try:
                    # Stage 1: nlpaug
                    speech_array = self.aug_stage1.augment(speech_array)
                    if isinstance(speech_array, list): speech_array = speech_array[0]
                    # Stage 2: audiomentations
                    speech_array = self.aug_stage2(samples=speech_array, sample_rate=16000)
                    
                    # 確保長度不超過 Whisper 的 30 秒 (480000 samples)
                    if len(speech_array) > 480000:
                        speech_array = speech_array[:480000]
                except Exception as e:
                    pass # 若增強失敗則使用原音

            features = self.processor.feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt").input_features[0]
            label_ids = self.processor.tokenizer(item["sentence"], max_length=448, truncation=True, return_tensors="pt").input_ids[0]
            
            input_features.append(features)
            labels.append(label_ids)
            
        if not input_features: return None
        return {"input_features": torch.stack(input_features), "labels": pad_sequence(labels, batch_first=True, padding_value=-100)}

# --------- 5. 監控回調 (保留原功能) ---------
class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, cache_obj):
        self.cache = cache_obj
        self.train_losses, self.eval_losses = [], []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs: self.train_losses.append((logs.get("epoch", 0), logs["loss"]))
            if "eval_loss" in logs: self.eval_losses.append((logs.get("epoch", 0), logs["eval_loss"]))
            if state.global_step % 50 == 0:
                mem = psutil.virtual_memory()
                info = self.cache.get_cache_info() if self.cache else "No Cache"
                print(f"Step {state.global_step}: RAM {mem.percent}%, Cache: {info}")
    def on_train_end(self, args, state, control, **kwargs):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            if self.train_losses:
                e, v = zip(*self.train_losses); plt.plot(e, v, label="Train Loss")
            if self.eval_losses:
                e, v = zip(*self.eval_losses); plt.plot(e, v, label="Eval Loss", linewidth=2)
            plt.legend(); plt.savefig(os.path.join(args.output_dir, "loss_curve_optimized.png"))
        except: pass

# --------- 6. 主程式 ---------
def main():
    args = parse_args()
    output_dir = setup_environment()
    start_time = datetime.now()
    
    global audio_cache
    audio_cache = LightweightAudioCache(max_size_gb=min(args.cache_size_gb, 50.0))

    train_ds = load_dataset("json", data_files={"train": args.train_json}, split="train")
    eval_ds = load_dataset("json", data_files={"test": args.eval_json}, split="test")

    processor = WhisperProcessor.from_pretrained(args.base_model, language="Chinese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    train_ds = train_ds.filter(lambda x: args.min_audio_len <= x["duration"] <= args.max_audio_len)
    eval_ds = eval_ds.filter(lambda x: args.min_audio_len <= x["duration"] <= args.max_audio_len)
    train_ds = train_ds.map(prepare_example_minimal, num_proc=16)
    eval_ds = eval_ds.map(prepare_example_minimal, num_proc=16)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=1000,
        logging_steps=50,
        save_steps=15000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        eval_strategy="steps",
        eval_steps=15000,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        optim="adamw_torch_fused"
    )

    # 分別為 訓練 和 評估 建立不同的 Data Collator
    train_collator = AugmentedDataCollator(processor, augment=True)
    eval_collator = AugmentedDataCollator(processor, augment=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=train_collator, # 預設使用訓練用 collator
        tokenizer=processor.tokenizer,
        callbacks=[MemoryMonitorCallback(audio_cache)]
    )
    
    # 這裡的小技巧：在評估時手動切換為 eval_collator
    trainer.data_collator = train_collator
    
    print(f"訓練開始時間: {start_time}")
    print(f"語音增強已啟動: VTLP, SNR, TimeStretch, Pitch, BackgroundNoise, RIR")
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    processor.save_pretrained(args.output_dir)
    print(f"訓練結束時間: {datetime.now()}")

if __name__ == "__main__":
    main()