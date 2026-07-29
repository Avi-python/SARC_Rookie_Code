import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# ===================== 路徑設定 =====================
model_path = "/mnt/usr1/ethan1017/MLHW3/baseline/checkpoint-156"
# 修改：更新為你指定的子資料夾路徑
audio_root = "/mnt/usr1/ethan1017/MLHW3/train/train"
# 修改：更改輸出檔名，避免覆蓋原本跑全量的結果
output_file = "/mnt/usr1/ethan1017/MLHW3/transcriptions_200.txt"

# ===================== 載入 fine-tuned 模型與 processor =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用裝置: {device}")

processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# ===================== 抓取並篩選音檔 =====================
all_audio_files = []
for root, _, files in os.walk(audio_root):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".flac")):
            all_audio_files.append(os.path.join(root, file))

# 新增：先排序，確保每次抓到的「前200筆」都是同樣的檔案 (依檔名順序)
all_audio_files.sort()

# 修改：切片列表，只取前 200 筆
target_files = all_audio_files[:200]

print(f"資料夾中總共有 {len(all_audio_files)} 個音檔。")
print(f"將只針對前 {len(target_files)} 筆進行推論。")

# ===================== 推論 =====================
with open(output_file, "w", encoding="utf-8") as fout:
    # 修改：這裡遍歷的是 target_files (前200筆) 而不是 all_audio_files
    for file_path in tqdm(target_files, desc="Processing top 200 audio files"):
        try:
            # 保留相對路徑，方便對照原始資料夾結構
            file_name = os.path.relpath(file_path, audio_root)

            # 讀取音訊
            speech_array, sr = torchaudio.load(file_path)
            
            # 重取樣至 16000Hz (Whisper 要求)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                speech_array = resampler(speech_array)
            
            # 轉為 numpy array 並移除多餘維度
            speech_array = speech_array.squeeze().numpy()

            # 處理 input features
            inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt").to(device)

            # 推論
            with torch.no_grad():
                predicted_ids = model.generate(inputs.input_features)

            # 解碼成文字
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # 寫入檔案 (格式: 檔名 \t 轉錄文字)
            fout.write(f"{file_name}\t{transcription}\n")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

print(f"✅ 前 200 筆推論完成，結果已儲存到 {output_file}")