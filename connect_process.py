import os
import random
import json
from pydub import AudioSegment

def load_jsonl(jsonl_file):
    """
    (修改) 讀取 jsonl 格式的 transcript，只儲存 text
    """
    mapping = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            # 簡化：只儲存 audio_filepath -> text
            mapping[entry["audio_filepath"]] = entry["text"]
    return mapping

def load_all_audio(wav_list):
    """
    (保留)
    將所有音檔讀入快取，避免重複讀檔
    """
    audio_cache = {}
    print(f"開始載入 {len(wav_list)} 個音檔到快取...")
    count = 0
    for wav in wav_list:
        try:
            audio_cache[wav] = AudioSegment.from_wav(wav)
            count += 1
            if count % 200 == 0:
                print(f"  已載入 {count}/{len(wav_list)}...")
        except Exception as e:
            print(f"警告：讀取 {wav} 失敗: {e}。將跳過此檔案。")
    print(f"快取載入完成，共 {len(audio_cache)} 個有效音檔。")
    return audio_cache

def concatenate_with_cache(
    wav_list_to_use,  #  4) 這是已排好序的列表
    transcript_map, 
    output_wav,
    audio_cache,      
    min_duration=5*1000, 
    max_duration=30*1000,
    min_silence_ms=100, #  3) 最小停頓
    max_silence_ms=500  #  3) 最大停頓
):
    
    target_duration = random.randint(min_duration, max_duration)

    combined = AudioSegment.silent(duration=0)
    total_duration = 0
    used_files = []
    combined_texts = []

    for wav_path in wav_list_to_use:
        # 確保 wav_path 真的在快取中 (避免 load_all_audio 時失敗的檔案)
        if wav_path not in audio_cache:
            continue
            
        audio = audio_cache[wav_path]
        
        # ---  3) 加入隨機靜音 ---
        # 只有在不是第一個音檔時才加入
        if total_duration > 0:
            silence_duration = random.randint(min_silence_ms, max_silence_ms)
            combined += AudioSegment.silent(duration=silence_duration)
            total_duration += silence_duration

        # 檢查是否會超過最大長度
        if total_duration + len(audio) > max_duration:
            # 如果加上這個音檔會太長，就停止
            # 但如果目前長度還沒到 min_duration，就強迫加這一個並結束
            if total_duration < min_duration:
                 pass # 強迫加入
            else:
                 break # 已經夠長了，停止

        combined += audio
        total_duration += len(audio)
        used_files.append(wav_path)
        
        # (修改) 使用簡化的 transcript_map
        if wav_path in transcript_map:
            combined_texts.append(transcript_map[wav_path])
        
        # 如果已經超過 target_duration，也停止
        if total_duration >= target_duration:
            break

    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    combined.export(output_wav, format="wav")
    
    #  5) 使用 " " (空格) 來 join 文本
    return used_files, total_duration / 1000, " ".join(combined_texts)

def generate_multiple_with_limits_fast(root_dir, transcript_file, output_dir,
                                       num_outputs=10, min_duration=5*1000,
                                       max_duration=30*1000, max_per_file=20,
                                       #  3) 傳入靜音參數
                                       min_silence_ms=100,
                                       max_silence_ms=500
                                      ):
    
    transcript_map = load_jsonl(transcript_file)
    print(f"成功讀取 {len(transcript_map)} 筆 transcript 紀錄。")

    wav_list = [os.path.join(root, f)
                for root, _, files in os.walk(root_dir)
                for f in files if f.endswith(".wav")]
    
    # 過濾掉不在 transcript_map 中的 wav (非常重要)
    original_count = len(wav_list)
    wav_list = [w for w in wav_list if w in transcript_map]
    print(f"找到 {original_count} 個 wav 檔，其中 {len(wav_list)} 個在 transcript 中。")

    if not wav_list:
        print("找不到任何 wav 檔！")
        return

    os.makedirs(output_dir, exist_ok=True)

    # (保留) 建立音訊快取
    audio_cache = load_all_audio(wav_list)
    
    # 再次過濾，只使用成功載入快取的檔案
    wav_list = [w for w in wav_list if w in audio_cache]
    print(f"快取載入成功，最終使用 {len(wav_list)} 個音檔進行拼接。")

    usage_count = {wav: 0 for wav in wav_list}
    jsonl_out = os.path.join(output_dir, "combined.jsonl")

    with open(jsonl_out, "w", encoding="utf-8") as fout:
        for i in range(num_outputs):
            
            # 篩選還沒超過 max_per_file 的音檔
            available_wavs = [w for w in wav_list if usage_count[w] < max_per_file]
            
            if not available_wavs:
                print(f"所有音檔都已達到 {max_per_file} 次使用上限，停止生成。")
                break

            #  4) 修正後的排序邏輯
            # 優先使用未使用過的
            unused_wavs = [w for w in available_wavs if usage_count[w] == 0]
            
            if unused_wavs:
                random.shuffle(unused_wavs)
                # 取得不在 unused_wavs 中的其他可用 wavs
                other_wavs = [w for w in available_wavs if usage_count[w] > 0]
                random.shuffle(other_wavs)
                # 確保 unused_wavs 在前面被優先挑選
                wavs_to_use = unused_wavs + other_wavs
            else:
                # 如果全都用過了，就全部隨機打亂
                random.shuffle(available_wavs)
                wavs_to_use = available_wavs

            output_wav = os.path.join(output_dir, f"combined_{i}.wav")
            
            used_files, dur, combined_text = concatenate_with_cache(
                wavs_to_use, transcript_map, output_wav, audio_cache,
                min_duration, max_duration,
                min_silence_ms, max_silence_ms #  3)
            )

            # 更新使用次數
            for f in used_files:
                usage_count[f] += 1

            record = {
                "audio_filepath": os.path.abspath(output_wav),
                "text": combined_text
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"✔ {output_wav} | {dur:.2f} 秒 | {len(used_files)} 段")

    print("\n音檔使用次數統計 (僅顯示有使用的)：")
    for f, count in usage_count.items():
        if count > 0:
            print(f"{os.path.basename(f)}: {count} 次")

generate_multiple_with_limits_fast(
    root_dir="/mnt/usr1/ethan1017/MLHW3/train/train",
    transcript_file="/mnt/usr1/ethan1017/MLHW3/train.jsonl",
    output_dir="/mnt/usr1/ethan1017/MLHW3/train/train2", 
    num_outputs=3100,
    min_duration=10*1000,
    max_duration=30*1000,
    max_per_file=16,
    min_silence_ms=150,  #  3) 模擬自然的停頓
    max_silence_ms=800  #  3) 模擬較長的句子間停頓
)