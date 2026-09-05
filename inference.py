#!/usr/bin/env python3
"""
Whisper Audio Inference (File & Microphone)
===========================================
支援音訊檔案 (例如 axr_avi.m4a / sxr_avi.m4a / wav 等) 與麥克風即時錄音辨識。

使用方式:
    # 1. 預設讀取音訊檔案 (自動尋找 axr_avi.m4a 或 sxr_avi.m4a)
    python inference.py

    # 2. 指定音訊檔案路徑
    python inference.py --audio_path axr_avi.m4a
    python inference.py --audio_path sxr_avi.m4a

    # 3. 指定模型 Checkpoint
    python inference.py --audio_path axr_avi.m4a --model_path ./checkpoint-165000-evaluation/checkpoint-165000

    # 4. 使用麥克風輸入
    python inference.py --mic
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration


def get_default_model_path():
    """尋找本機預設的模型路徑，若無則使用 OpenAI 官方模型"""
    candidates = [
        "/mnt/usr1/avi/Projects/whisper_code/checkpoint-165000-evaluation/checkpoint-165000",
        "./checkpoint-165000-evaluation/checkpoint-165000",
        "/mnt/usr1/avi/Projects/whisper_code/whisper-finetuned-formosan-v1/checkpoint-69000",
        "./whisper-finetuned-formosan-v1/checkpoint-69000",
        "openai/whisper-large-v3-turbo"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "openai/whisper-large-v3-turbo"


def resolve_audio_path(requested_path):
    """解析音訊路徑，若 axr_avi.m4a 不存在但 sxr_avi.m4a 存在則自動對應"""
    if requested_path and os.path.exists(requested_path):
        return requested_path

    candidates = [
        requested_path,
        "axr_avi.m4a",
        "sxr_avi.m4a",
        os.path.join(os.path.dirname(__file__), "axr_avi.m4a"),
        os.path.join(os.path.dirname(__file__), "sxr_avi.m4a"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            if requested_path and c != requested_path:
                print(f"[提示] 未找到指定檔案 '{requested_path}'，已自動使用找到的音檔: '{c}'")
            return c
    return requested_path


def load_audio_file(file_path, target_sr=16000):
    """讀取音訊檔案 (支援 m4a, wav, mp3, flac 等) 並重取樣至 target_sr"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到音訊檔案: {file_path}")

    # 方法 1: 使用 torchaudio 讀取 (支援 m4a / ffmpeg 後端)
    try:
        import torchaudio
        speech_array, sr = torchaudio.load(file_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            speech_array = resampler(speech_array)
        # 若為多聲道則轉換為單聲道
        if speech_array.ndim > 1 and speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
        return speech_array.squeeze().numpy().astype(np.float32)
    except Exception as err_torchaudio:
        last_error = err_torchaudio

    # 方法 2: 使用 soundfile 讀取
    try:
        import soundfile as sf
        data, sr = sf.read(file_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            tensor_data = torch.from_numpy(data).unsqueeze(0)
            data = resampler(tensor_data).squeeze().numpy()
        return data.astype(np.float32)
    except Exception as err_sf:
        last_error = err_sf

    # 方法 3: 使用 pydub 讀取
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(file_path)
        seg = seg.set_frame_rate(target_sr).set_channels(1)
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        max_val = float(1 << (seg.sample_width * 8 - 1))
        samples /= max_val
        return samples
    except Exception as err_pydub:
        last_error = err_pydub

    raise RuntimeError(f"無法讀取音訊檔案 '{file_path}'，錯誤原因: {last_error}")


def record_audio_manual(sample_rate=16000, device=None):
    """麥克風手動模式：按 Enter 開始錄音，再按 Enter 結束錄音"""
    try:
        import sounddevice as sd
    except ImportError:
        print("[錯誤] 使用麥克風錄音需安裝 sounddevice 套件：pip install sounddevice")
        sys.exit(1)

    print("\n👉 請按 [Enter] 鍵開始錄音...")
    sys.stdin.readline()

    print("🎙️  [錄音中...] 請開始說話，說完後請再按一次 [Enter] 鍵結束錄音...")
    audio_chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[音訊狀態警告] {status}", file=sys.stderr)
        audio_chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", device=device, callback=callback):
        sys.stdin.readline()

    print("⏹️  [錄音結束]")
    if not audio_chunks:
        return np.array([], dtype=np.float32)

    audio_data = np.concatenate(audio_chunks, axis=0).squeeze()
    return audio_data


def record_audio_fixed(duration=5.0, sample_rate=16000, device=None):
    """麥克風固定時長模式：倒數並錄製指定秒數"""
    try:
        import sounddevice as sd
    except ImportError:
        print("[錯誤] 使用麥克風錄音需安裝 sounddevice 套件：pip install sounddevice")
        sys.exit(1)

    print(f"\n👉 請按 [Enter] 鍵開始錄音 (將錄製 {duration} 秒)...")
    sys.stdin.readline()

    print(f"🎙️  [錄音中...] 請開始說話 (倒數 {duration} 秒)...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device
    )
    sd.wait()
    print("⏹️  [錄音結束]")
    return recording.squeeze()


def list_audio_devices():
    """列出可用麥克風輸入裝置"""
    try:
        import sounddevice as sd
        print("\n可用的音訊裝置列表：")
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                print(f"  [{idx}] {dev['name']} (輸入聲道數: {dev['max_input_channels']}, 預設取樣率: {dev['default_samplerate']}Hz)")
        print()
    except Exception as e:
        print(f"[錯誤] 查詢音訊裝置失敗: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper 音訊推論程式 (支援 m4a/wav 音檔與麥克風)")
    parser.add_argument(
        "--audio_path",
        type=str,
        default="axr_avi.m4a",
        help="輸入音訊檔案路徑 (預設: axr_avi.m4a，若不存在會自動尋找 sxr_avi.m4a)"
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="使用麥克風即時錄音模式 (若未指定且找不到音檔時亦會提示)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=get_default_model_path(),
        help="Whisper 模型 checkpoint 路徑或 HuggingFace 模型名稱"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="若 checkpoint 中缺少 processor 時備用的 base model 名稱"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="取樣率 (Whisper 預設為 16000 Hz)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="麥克風模式下的固定錄音秒數 (若未設定，則按 Enter 開始/結束錄音)"
    )
    parser.add_argument(
        "--device_index",
        type=int,
        default=None,
        help="麥克風輸入裝置編號 (可搭配 --list_devices 查詢)"
    )
    parser.add_argument(
        "--list_devices",
        action="store_true",
        help="列出所有可用的音訊輸入裝置並退出"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="強制指定語言代碼 (如 zh, en)，若未設定則由模型自動判定"
    )
    return parser.parse_args()


def transcribe_audio(audio_data, model, processor, device, sample_rate=16000, language=None):
    """執行特徵轉換與模型解碼推論"""
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").to(device)

    gen_kwargs = {}
    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        gen_kwargs["forced_decoder_ids"] = forced_decoder_ids

    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features, **gen_kwargs)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def main():
    args = parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # 1. 檢查並設定推論硬體裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用推論裝置: {device}")

    # 2. 載入 Processor
    print(f"[INFO] 正在載入 Processor: {args.model_path} ...")
    try:
        processor = WhisperProcessor.from_pretrained(args.model_path)
    except Exception:
        print(f"[INFO] 改由 base model ({args.base_model}) 載入 Processor...")
        processor = WhisperProcessor.from_pretrained(args.base_model)

    # 3. 載入 Model
    print(f"[INFO] 正在載入 Whisper 模型: {args.model_path} ...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.eval()
    print("[INFO] 模型載入完成！\n" + "=" * 60)

    # 4. 判斷執行模式：音訊檔案推論 或 麥克風錄音推論
    target_audio_file = None
    if not args.mic:
        target_audio_file = resolve_audio_path(args.audio_path)

    # --- 模式 A: 音訊檔案推論 ---
    if target_audio_file and os.path.exists(target_audio_file):
        print(f"🎵 [檔案推論模式] 正在讀取音訊檔案: {target_audio_file}")
        t0 = time.time()
        audio = load_audio_file(target_audio_file, target_sr=args.sample_rate)
        duration_sec = len(audio) / args.sample_rate
        print(f"[音訊資訊] 長度: {duration_sec:.2f} 秒 (取樣點數: {len(audio)}, 取樣率: {args.sample_rate}Hz)")

        print("🔄 [辨識中] 正在進行語音轉錄...")
        transcription = transcribe_audio(
            audio_data=audio,
            model=model,
            processor=processor,
            device=device,
            sample_rate=args.sample_rate,
            language=args.language
        )
        elapsed = time.time() - t0

        print("=" * 60)
        print(f"📁 音檔名稱: {target_audio_file}")
        print(f"⏱️  耗時: {elapsed:.2f} 秒")
        print(f"📝 【轉錄結果】:\n{transcription}")
        print("=" * 60)
        return

    # 若指定檔案不存在且未開 --mic
    if not args.mic:
        print(f"[錯誤] 找不到音訊檔案: '{args.audio_path}'。")
        print("若要使用麥克風錄音，請加上 --mic 參數；或者請確認音檔路徑是否正確。")
        sys.exit(1)

    # --- 模式 B: 麥克風即時錄音推論 ---
    print("🎙️ [麥克風推論模式]")
    print("💡 提示：輸入 'q' 並按 Enter 可退出程式。直接按 Enter 可繼續錄音。")

    while True:
        try:
            print("\n準備進行下一次錄音 (按 q 退出，或按 Enter 繼續): ", end="", flush=True)
            user_choice = sys.stdin.readline().strip().lower()
            if user_choice == "q":
                print("\n[INFO] 已結束程式。")
                break

            if args.duration is not None and args.duration > 0:
                audio = record_audio_fixed(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    device=args.device_index
                )
            else:
                audio = record_audio_manual(
                    sample_rate=args.sample_rate,
                    device=args.device_index
                )

            if audio is None or len(audio) == 0:
                print("[警告] 未錄到任何音訊，請重試。")
                continue

            max_amp = np.max(np.abs(audio)) if len(audio) > 0 else 0
            audio_duration = len(audio) / args.sample_rate
            print(f"[音訊資訊] 長度: {audio_duration:.2f} 秒, 最大振幅: {max_amp:.4f}")

            if max_amp < 0.005:
                print("⚠️  [注意] 錄音音量過小，請確認麥克風是否靜音或輸入設備是否正確。")

            print("🔄 [辨識中] 正在進行語音轉錄...")
            transcription = transcribe_audio(
                audio_data=audio,
                model=model,
                processor=processor,
                device=device,
                sample_rate=args.sample_rate,
                language=args.language
            )

            print("-" * 60)
            print(f"📝 【轉錄結果】: {transcription}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n[INFO] 偵測到 Ctrl+C，程式已中斷。")
            break
        except Exception as e:
            print(f"\n[錯誤] 處理音訊時發生異常: {e}")


if __name__ == "__main__":
    main()
