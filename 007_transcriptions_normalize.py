import re

def normalize_text(text):
    # 移除標點符號（保留特殊符號、拼音、數字）
    text = re.sub(r'[，。！？、．：；《》……––「」,.!?，。！？；：「」『』、（）〈〉《》【】]', '', text)
    
    # 對中文漢字加空格，保留原本空格、數字、拼音與特殊符號
    def add_space(match):
        return ' '.join(match.group(0))  # 每個漢字加空格

    # Unicode 範圍 \u4e00-\u9fff 為中文漢字
    text = re.sub(r'[\u4e00-\u9fff]+', add_space, text)
    
    # 將多個空格縮成一個
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_file(input_file, output_file):
    results = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split('\t', 1)  # 你的例子是用 tab 分隔
            if len(parts) == 2:
                filename, transcript = parts
                transcript_norm = normalize_text(transcript)
                results.append(f"{filename} {transcript_norm}")
            else:
                results.append(line.strip())
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for res in results:
            outfile.write(res + "\n")
    
    print(f"處理完成！結果已儲存至 {output_file}")

# 設定輸入和輸出文件
input_txt = "/mnt/usr1/ethan1017/MLHW3/transcriptions_200.txt"
output_txt = "/mnt/usr1/ethan1017/MLHW3/hyp.txt"

process_file(input_txt, output_txt)



