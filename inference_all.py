# %%
import os
import io
import glob
import torch
import csv
import argparse
import torchaudio
import numpy as np
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# %%
# ===================== 路徑與參數設定 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Batch Inference with Language Filter")
    parser.add_argument("--language_group", "--language_id", dest="language_group", type=str, default="dru",
                        help="Target language ID group (e.g. 'dru', 'dru,tsu'). Set to 'all' for all languages.")
    parser.add_argument("--root_dir", type=str, default="/nfs/DGX2/Workspace/avi/processed_data_independent3/",
                        help="Parquet dataset root directory")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model directory or checkpoint path")
    parser.add_argument("--base_model_name", type=str, default="openai/whisper-large-v3-turbo",
                        help="Fallback base model for processor")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output CSV path (default: transcriptions_<language_group>.csv)")
    parser.add_argument("--target_language", type=str, default="id",
                        help="Whisper target decoding language code (default: 'id')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to infer (default: None for all)")
    parsed = parser.parse_known_args()[0]

    group_slug = parsed.language_group.replace(",", "_").replace(" ", "_") if parsed.language_group and parsed.language_group.lower() != "all" else "all"

    # 自動設定預設模型路徑
    if parsed.model_path is None:
        finetuned_group_model = f"/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-{group_slug}/final_model"
        formosan_v2_model = "/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-v2/final_model-evaluation/final_model"
        if os.path.exists(finetuned_group_model):
            parsed.model_path = finetuned_group_model
        elif os.path.exists(formosan_v2_model):
            parsed.model_path = formosan_v2_model
        else:
            parsed.model_path = parsed.base_model_name

    # 自動設定預設輸出檔名
    if parsed.output_file is None:
        if group_slug != "all":
            parsed.output_file = f"/mnt/usr1/avi/Projects/whisper_code/transcriptions_{group_slug}.csv"
        else:
            parsed.output_file = "/mnt/usr1/avi/Projects/whisper_code/transcriptions.csv"

    return parsed


args = parse_args()
model_path = args.model_path
base_model_name = args.base_model_name
root_dir = args.root_dir
output_file = args.output_file
max_samples = args.max_samples
target_language = args.target_language
language_group = args.language_group


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


# %%
# ===================== 載入 fine-tuned 模型與 processor =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用裝置: {device}")
print(f"篩選目標語言組 (language_group): '{language_group}'")
print(f"輸出目標檔案 (output_file): {output_file}")

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
print(f"正在搜尋 {root_dir} 下的 Parquet 檔案 (language_group: '{language_group}')...")
all_parquet_files = glob.glob(f"{root_dir}/**/split=test/**/*.parquet", recursive=True)
parquet_files = filter_parquet_files(all_parquet_files, language_group, root_dir)
print(f"共找到 {len(parquet_files)} 個符合的 parquet 檔案 (總共 {len(all_parquet_files)} 個)。")

if not parquet_files:
    raise FileNotFoundError(f"找不到符合 language_group '{language_group}' 的 parquet 檔案於目錄: {root_dir}")

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