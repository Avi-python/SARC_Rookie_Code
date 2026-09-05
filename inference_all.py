# %%
import os
import io
import glob
import torch
import csv
import torchaudio
import numpy as np
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# %%
# ===================== 路徑與參數設定 =====================
# 模型路徑 (支援 Projects 與非 Projects 路徑)
model_path = "/mnt/usr1/avi/Projects/whisper_code/checkpoint-165000-evaluation/checkpoint-165000"

base_model_name = "openai/whisper-large-v3-turbo"

# Parquet 資料集目錄 (參照 train_test.py lines 82-94)
root_dir = "/nfs/DGX2/Workspace/avi/processed_data_independent3/"

# 輸出推論結果檔名
output_file = "/mnt/usr1/avi/Projects/whisper_code/transcriptions.csv"

# 限制推論筆數 (例如前 200 筆；若要跑選定 parquet 的全量可設為 None)
max_samples = None

# 推論目標語言 (預設: 印尼語 'id')
target_language = "id"

# %%
# ===================== 載入 fine-tuned 模型與 processor =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用裝置: {device}")

try:
    processor = WhisperProcessor.from_pretrained(model_path, language=target_language, task="transcribe")
except Exception:
    print(f"從 checkpoint 載入 processor 失敗，改由 base model ({base_model_name}) 載入...")
    processor = WhisperProcessor.from_pretrained(base_model_name, language=target_language, task="transcribe")

print(f"正在載入模型: {model_path} ...")
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()
forced_decoder_ids = processor.get_decoder_prompt_ids(language=target_language, task="transcribe")
print("模型載入成功！")

# %%
# ===================== 搜尋並載入 Parquet 資料集 =====================
print(f"正在搜尋 {root_dir} 下的 Parquet 檔案...")
parquet_files = sorted(glob.glob(f"{root_dir}/**/split=test/**/*.parquet", recursive=True))
print(f"共找到 {len(parquet_files)} 個 parquet 檔案。")

if not parquet_files:
    raise FileNotFoundError(f"找不到任何 parquet 檔案於目錄: {root_dir}")

print("正在依語言目錄分類並載入 Parquet 資料集...")
files_by_lang = defaultdict(list)
for pfile in parquet_files:
    # 語言標籤位於 root_dir 的下一層子目錄
    rel_path = os.path.relpath(pfile, root_dir)
    raw_lang_id = rel_path.split(os.sep)[0]
    # 格式化 lang_id (去除可能誤加的單雙引號與前後空白，如 'trv_Delu Valley' -> trv_Delu Valley)
    lang_id = raw_lang_id.replace("'", "").replace('"', '').strip()
    files_by_lang[lang_id].append(pfile)

print(f"共找到 {len(files_by_lang)} 個語言類別: {list(files_by_lang.keys())}")

datasets_list = []
for lang_id, files in sorted(files_by_lang.items()):
    ds = load_dataset("parquet", data_files={"test": files}, split="test")
    ds = ds.add_column("language_id", [lang_id] * len(ds))
    datasets_list.append(ds)

dataset = concatenate_datasets(datasets_list)
print(f"資料集載入成功！該 Parquet 總筆數: {len(dataset)}")
print(f"資料集欄位: {dataset.column_names}")

# 篩選部分資料 (如前 max_samples 筆)
if max_samples is not None and max_samples < len(dataset):
    dataset = dataset.select(range(max_samples))
    print(f"將只針對前 {len(dataset)} 筆進行推論。")
else:
    print(f"將針對全部 {len(dataset)} 筆進行推論。")

# %%
# ===================== 推論 =====================
sr_target = 16000

with open(output_file, "w", encoding="utf-8", newline="") as fout:
    writer = csv.writer(fout)
    
     # 寫入表頭 (包含 language_id)
    writer.writerow(["index", "language_id", "audio_size", "ground_truth", "prediction"])

    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {len(dataset)} audio samples")):
        try:
            # 1. 提取音訊 raw bytes (格式參照 train_test.py lines 147-148)
            audio_bytes = example["audio_bytes"]
            audio_bytes = np.array(audio_bytes, dtype=np.int8).view(np.uint8).tobytes()

            # 2. 從記憶體中讀取音訊
            speech_array, sr_orig = torchaudio.load(io.BytesIO(audio_bytes))

            # 3. 重取樣至 16000Hz (Whisper 要求)
            if sr_orig != sr_target:
                resampler = torchaudio.transforms.Resample(sr_orig, sr_target)
                speech_array = resampler(speech_array)

            # 轉為 numpy array 並移除多餘維度
            speech_array = speech_array.squeeze().numpy()

            # 4. 處理特徵
            inputs = processor(speech_array, sampling_rate=sr_target, return_tensors="pt").to(device)

            # 5. 推論
            with torch.no_grad():
                predicted_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)

            # 6. 解碼成文字
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            ground_truth = example.get("text", "")
            audio_size = example.get("audio_size", "")
            lang_id = example.get("language_id", "")

            # 寫入檔案
            writer.writerow([idx, lang_id, audio_size, ground_truth, transcription])

        except Exception as e:
            print(f"Error processing sample index {idx}: {e}")
            writer.writerow([idx, example.get("language_id", ""), example.get("audio_size", ""), example.get("text", ""), f"ERROR: {e}"])

print(f"✅ 推論完成，共處理 {len(dataset)} 筆資料，結果已儲存到 {output_file}")