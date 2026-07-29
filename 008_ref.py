import json
import re

input_jsonl = "/mnt/usr1/ethan1017/MLHW3/train2.jsonl"
output_txt = "/mnt/usr1/ethan1017/MLHW3/ref.txt"

prefix_to_remove = "/mnt/usr1/ethan1017/MLHW3/train/train/"

def normalize_text(text):
    # 移除標點符號（保留拼音、數字、特殊符號）
    text = re.sub(r'[，。！？、．：；《》……––「」,.!?，。！？；：「」『』、（）〈〉《》【】]', '', text)
    
    # 對中文漢字加空格，保留原本空格、拼音與數字
    def add_space(match):
        return ' '.join(match.group(0))  # 每個漢字加空格
    
    text = re.sub(r'[\u4e00-\u9fff]+', add_space, text)
    
    # 將多個空格縮成一個
    text = re.sub(r'\s+', ' ', text).strip()
    return text

with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_txt, "w", encoding="utf-8") as fout:
    skipped = 0  # 記錄跳過的空文字行
    for line in fin:
        data = json.loads(line)
        text = data.get("text", "").strip()  # 取文字並去掉前後空白
        if text == "":
            skipped += 1
            continue  # 跳過空文字
        
        # 移除路徑前綴
        audio_path = data["audio_filepath"].replace(prefix_to_remove, "")
        # 文字正規化
        text_norm = normalize_text(text)
        
        fout.write(f"{audio_path} {text_norm}\n")

print(f"✅ 已儲存正解檔為：{output_txt}")
if skipped > 0:
    print(f"⚠️ 跳過 {skipped} 行空文字。")






