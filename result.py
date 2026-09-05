import os
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from tqdm import tqdm
import csv  # 從 normalize.py 匯入

# --- 全局設定 (來自兩個檔案) ---
model_path = "/mnt/usr1/ethan1017/MLHW3/baseline/checkpoint-1866"
audio_folder = "/mnt/usr1/ethan1017/MLHW3/test"
# 直接使用 normalize.py 的最終輸出路徑
output_file = "/mnt/usr1/ethan1017/MLHW3/submission4.csv"

# --- 載入模型與處理器 (來自 inference.py) ---
try:
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ 成功載入本地端的 processor 與 tokenizer。")
except Exception as e:
    print(f"⚠️ 載入本地 processor 失敗 ({e})，嘗試從 'openai/whisper-base' 重新下載...")
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    processor.save_pretrained(model_path)
    print("✅ Tokenizer 已下載並儲存到 model_path。")

# 載入模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
model.eval()
print(f"✅ 模型已成功載入到 {device} 裝置。")


def transcribe(audio_path):
    """
    (來自 inference.py)
    使用強制的 "zh" 解碼來轉錄音檔。
    返回原始轉錄結果。
    """
    try:
        speech_array, sr = torchaudio.load(audio_path)
        # 檢查並重採樣至 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech_array = resampler(speech_array)
        speech_array = speech_array.squeeze().numpy()

        # 提取特徵
        input_features = processor.feature_extractor(speech_array, sampling_rate=16000).input_features
        input_features = torch.tensor(input_features).to(device)

        # 強制使用印尼語進行轉錄
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="id", task="transcribe")

        # 執行推論
        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, num_beams=10)

        # 解碼
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    except Exception as e:
        print(f"❌ 轉錄失敗: {audio_path} | 錯誤訊息: {e}")
        return None

def transcribe_and_normalize_folder(root_folder, output_path):
    """
    (整合 inference.py 的 'transcribe_folder' 和 normalize.py 的 'main' 邏輯)
    1. 遞迴尋找所有 .wav 檔案。
    2. 逐一進行轉錄。
    3. 即時正規化 ID (檔名)。
    4. 將 (ID, sentence) 寫入最終的 CSV 檔。
    """
    all_audio_files = []

    # 1. 遞迴尋找所有 .wav 檔案 (來自 inference.py)
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".wav"):
                all_audio_files.append(os.path.join(root, file))

    if not all_audio_files:
        print(f"⚠️ 在 {root_folder} 中找不到任何 .wav 檔案。")
        return

    print(f"🔄 準備開始轉錄 {len(all_audio_files)} 個音檔...\n")

    # 4. 開啟最終的 CSV 檔準備寫入 (來自 normalize.py)
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "sentence"])  # 寫入表頭

        # 2. 逐一進行轉錄 (來自 inference.py)
        for audio_path in tqdm(all_audio_files, desc="Processing"):
            transcription = transcribe(audio_path)
            
            if transcription:
                # 3. 即時正規化 ID (來自 normalize.py 的邏輯)
                # 使用 os.path.basename 擷取檔名，比 split('/') 更安全
                wav_id = os.path.basename(audio_path).replace(".wav", "")
                
                # 4. 將 (ID, sentence) 寫入 CSV (來自 normalize.py)
                writer.writerow([wav_id, transcription])
            else:
                # 處理轉錄失敗的檔案
                wav_id = os.path.basename(audio_path).replace(".wav", "")
                writer.writerow([wav_id, "TRANSCRIPTION_FAILED"])


if __name__ == "__main__":
    transcribe_and_normalize_folder(audio_folder, output_file)
    # 使用 normalize.py 的最終完成訊息
    print(f"✅ 轉換完成，已輸出到 {output_file}")