import jiwer
from difflib import SequenceMatcher
import os

def normalize_text(text):
    """
    Normalize input text to lowercase (mainly for English).
    Keeps other characters (e.g., Chinese) unchanged.
    """
    return text.lower()

def highlight_differences(ref, hyp):
    """
    Compare reference and hypothesis and highlight differences.
    - Deletion: [DEL: word]
    - Substitution: [SUB: original -> new]
    - Insertion: [INS: word]
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    matcher = SequenceMatcher(None, ref_words, hyp_words)
    highlighted_ref = []
    highlighted_hyp = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            highlighted_ref.extend(ref_words[i1:i2])
            highlighted_hyp.extend(hyp_words[j1:j2])
        elif tag == "replace":
            for rw, hw in zip(ref_words[i1:i2], hyp_words[j1:j2]):
                highlighted_ref.append(f"[SUB: {rw} -> {hw}]")
                highlighted_hyp.append(f"[SUB: {hw}]")
        elif tag == "delete":
            for rw in ref_words[i1:i2]:
                highlighted_ref.append(f"[DEL: {rw}]")
        elif tag == "insert":
            for hw in hyp_words[j1:j2]:
                highlighted_hyp.append(f"[INS: {hw}]")
                
    return " ".join(highlighted_ref), " ".join(highlighted_hyp)

def load_data_by_id(file_path):
    """
    Load text data from file and return as a dict: {id: content}.
    Each line should start with an ID followed by content.
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    news_id, content = parts
                else:
                    news_id, content = parts[0], ""
                data[news_id] = content
    return data

def calculate_wer_with_highlights_by_id(reference_file, hypothesis_file, output_file="/mnt/usr1/ethan1017/MLHW3/mer_results_before.txt"):
    # Load reference and hypothesis data
    ref_data = load_data_by_id(reference_file)
    hyp_data = load_data_by_id(hypothesis_file)
    
    common_ids = set(ref_data.keys()) & set(hyp_data.keys())
    missing_in_hyp = set(ref_data.keys()) - set(hyp_data.keys())
    missing_in_ref = set(hyp_data.keys()) - set(ref_data.keys())
    
    if missing_in_hyp:
        print("Warning: The following reference IDs are missing in hypothesis:", ", ".join(missing_in_hyp))
    if missing_in_ref:
        print("Warning: The following hypothesis IDs are missing in reference:", ", ".join(missing_in_ref))
    
    total_errors = 0
    total_words = 0
    total_del = 0
    total_ins = 0
    total_sub = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("=== Per-sample WER results ===\n\n")
        
        for news_id in sorted(common_ids):
            ref_text = ref_data[news_id]
            hyp_text = hyp_data[news_id]
            
            ref_norm = normalize_text(ref_text)
            hyp_norm = normalize_text(hyp_text)
            
            # Fix: 使用 process_words 取代 compute_measures，並支援 jiwer 3.x
            # 注意：這裡回傳的是一個 output 物件，不是字典
            output = jiwer.process_words(ref_norm, hyp_norm)
            
            wer = output.wer
            substitutions = output.substitutions
            deletions = output.deletions
            insertions = output.insertions
            
            # 如果需要手動計算 word count (分母)，通常是 len(ref)
            # jiwer 的 output 物件裡也有，但為了保持你原本邏輯一致，維持原本寫法：
            word_count = len(ref_norm.split())
            
            total_errors += substitutions + deletions + insertions
            total_words += word_count
            total_del += deletions
            total_ins += insertions
            total_sub += substitutions
            
            highlighted_ref, highlighted_hyp = highlight_differences(ref_norm, hyp_norm)
            
            out_f.write(f"-- ID: {news_id}\n")
            out_f.write(f"    Reference: {ref_text}\n")
            out_f.write(f"    Hypothesis: {hyp_text}\n")
            out_f.write(f"    WER: {wer:.4f}\n")
            out_f.write(f"    Deletions (D): {deletions}\n")
            out_f.write(f"    Substitutions (S): {substitutions}\n")
            out_f.write(f"    Insertions (I): {insertions}\n")
            out_f.write(f"    Word Count: {word_count}\n")
            out_f.write(f"    Highlighted Differences:\n")
            out_f.write(f"        Ref: {highlighted_ref}\n")
            out_f.write(f"        Hyp: {highlighted_hyp}\n\n")
        
        # Summary
        overall_wer = total_errors / total_words if total_words > 0 else 0
        overall_del = total_del / total_words if total_words > 0 else 0
        overall_ins = total_ins / total_words if total_words > 0 else 0
        overall_sub = total_sub / total_words if total_words > 0 else 0
        
        out_f.write("=== Overall WER Summary ===\n")
        out_f.write(f"   Overall WER: {overall_wer:.4f}\n")
        out_f.write(f"   Deletion Rate: {overall_del:.4f}\n")
        out_f.write(f"   Insertion Rate: {overall_ins:.4f}\n")
        out_f.write(f"   Substitution Rate: {overall_sub:.4f}\n")
        out_f.write(f"   Total Words: {total_words}\n")
    
    print(f"\nWER evaluation complete. Results saved to {output_file}")
    return overall_wer

# Example usage
reference_txt = "/mnt/usr1/ethan1017/MLHW3/ref.txt"
hypothesis_txt = "/mnt/usr1/ethan1017/MLHW3/hyp.txt"

# 檢查檔案是否存在，避免路徑錯誤
if os.path.exists(reference_txt) and os.path.exists(hypothesis_txt):
    calculate_wer_with_highlights_by_id(reference_txt, hypothesis_txt)
else:
    print(f"Error: 找不到輸入檔案。\n請確認 {reference_txt} 與 {hypothesis_txt} 路徑正確。")