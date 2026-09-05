import csv
import os
import jiwer
from collections import defaultdict
from difflib import SequenceMatcher

def normalize_text(text):
    """
    Normalize input text to lowercase and trim extra whitespace.
    """
    if text is None:
        return ""
    return str(text).lower().strip()

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
            rw_slice = ref_words[i1:i2]
            hw_slice = hyp_words[j1:j2]
            min_len = min(len(rw_slice), len(hw_slice))
            for rw, hw in zip(rw_slice[:min_len], hw_slice[:min_len]):
                highlighted_ref.append(f"[SUB: {rw} -> {hw}]")
                highlighted_hyp.append(f"[SUB: {hw}]")
            for rw in rw_slice[min_len:]:
                highlighted_ref.append(f"[DEL: {rw}]")
            for hw in hw_slice[min_len:]:
                highlighted_hyp.append(f"[INS: {hw}]")
        elif tag == "delete":
            for rw in ref_words[i1:i2]:
                highlighted_ref.append(f"[DEL: {rw}]")
        elif tag == "insert":
            for hw in hyp_words[j1:j2]:
                highlighted_hyp.append(f"[INS: {hw}]")
                
    return " ".join(highlighted_ref), " ".join(highlighted_hyp)

def evaluate_transcriptions_csv(
    input_csv="/mnt/usr1/avi/Projects/whisper_code/transcriptions.csv",
    details_csv="/mnt/usr1/avi/Projects/whisper_code/eval_details.csv",
    results_csv="/mnt/usr1/avi/Projects/whisper_code/eval_results.csv"
):
    """
    Load transcriptions from input_csv (columns: index, language_id, ground_truth, prediction)
    Compute WER, CER, and detailed metrics using jiwer,
    save per-sample detailed results to details_csv (eval_details.csv),
    and save overall evaluation metrics to results_csv (eval_results.csv).
    """
    if not os.path.exists(input_csv):
        print(f"Error: 找不到輸入檔案: {input_csv}")
        return

    all_refs = []
    all_hyps = []
    lang_refs = defaultdict(list)
    lang_hyps = defaultdict(list)
    rows_to_write = []

    print(f"正在讀取 {input_csv} ...")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("index") if row.get("index") is not None else row.get("idx", "")
            lang_id = row.get("language_id", "") or ""
            ref = row.get("ground_truth", "") or ""
            hyp = row.get("prediction", "") or ""
            audio_size = row.get("audio_size", "") or ""

            # 若推論階段有錯誤訊息
            if hyp.startswith("ERROR:"):
                rows_to_write.append({
                    "idx": idx,
                    "language_id": lang_id,
                    "audio_size": audio_size,
                    "Reference": ref,
                    "Hypothesis": hyp,
                    "WER": "N/A",
                    "CER": "N/A",
                    "Deletions": "N/A",
                    "Substitutions": "N/A",
                    "Insertions": "N/A",
                    "Highlighted Differences": "ERROR in prediction"
                })
                continue

            ref_norm = normalize_text(ref)
            hyp_norm = normalize_text(hyp)

            all_refs.append(ref_norm)
            all_hyps.append(hyp_norm)
            if lang_id:
                lang_refs[lang_id].append(ref_norm)
                lang_hyps[lang_id].append(hyp_norm)

            # 計算單筆 WER 與 CER
            try:
                if not ref_norm and not hyp_norm:
                    wer = 0.0
                    cer = 0.0
                    substitutions = 0
                    deletions = 0
                    insertions = 0
                elif not ref_norm:
                    wer = 1.0
                    cer = 1.0
                    substitutions = 0
                    deletions = 0
                    insertions = len(hyp_norm.split())
                else:
                    # Word-level measures
                    word_out = jiwer.process_words(ref_norm, hyp_norm)
                    wer = word_out.wer
                    substitutions = word_out.substitutions
                    deletions = word_out.deletions
                    insertions = word_out.insertions

                    # Character-level measures
                    char_out = jiwer.process_characters(ref_norm, hyp_norm)
                    cer = char_out.cer
            except Exception as e:
                wer = "N/A"
                cer = "N/A"
                substitutions = "N/A"
                deletions = "N/A"
                insertions = "N/A"

            highlighted_ref, highlighted_hyp = highlight_differences(ref_norm, hyp_norm)
            diff_str = f"Ref: {highlighted_ref} | Hyp: {highlighted_hyp}" if (highlighted_ref != highlighted_hyp) else "Match"

            rows_to_write.append({
                "idx": idx,
                "language_id": lang_id,
                "audio_size": audio_size,
                "Reference": ref,
                "Hypothesis": hyp,
                "WER": f"{wer:.4f}" if isinstance(wer, (int, float)) else str(wer),
                "CER": f"{cer:.4f}" if isinstance(cer, (int, float)) else str(cer),
                "Deletions": deletions,
                "Substitutions": substitutions,
                "Insertions": insertions,
                "Highlighted Differences": diff_str
            })

    # 寫入個別詳細評估結果 CSV
    fieldnames = [
        "idx",
        "language_id",
        "audio_size",
        "Reference",
        "Hypothesis",
        "WER",
        "CER",
        "Deletions",
        "Substitutions",
        "Insertions",
        "Highlighted Differences"
    ]

    with open(details_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"✅ 個別詳細評估結果已儲存至: {details_csv}")

    # 計算整體 (Corpus-level) 與分語言 WER / CER，並寫入 eval_results.csv
    overall_rows = []
    if all_refs and all_hyps:
        overall_word_out = jiwer.process_words(all_refs, all_hyps)
        overall_char_out = jiwer.process_characters(all_refs, all_hyps)

        print("\n" + "=" * 35 + " Overall Summary " + "=" * 35)
        print(f"   Total Samples Processed: {len(all_refs)}")
        print(f"   Overall WER            : {overall_word_out.wer:.4f} ({overall_word_out.wer * 100:.2f}%)")
        print(f"   Overall CER            : {overall_char_out.cer:.4f} ({overall_char_out.cer * 100:.2f}%)")
        print(f"   Word Substitutions (S) : {overall_word_out.substitutions}")
        print(f"   Word Deletions (D)     : {overall_word_out.deletions}")
        print(f"   Word Insertions (I)    : {overall_word_out.insertions}")
        print(f"   Word Hits (H)          : {overall_word_out.hits}")
        print("=" * 87 + "\n")

        overall_rows.append({
            "language_id": "OVERALL",
            "total_samples": len(all_refs),
            "wer": f"{overall_word_out.wer:.4f}",
            "cer": f"{overall_char_out.cer:.4f}",
            "substitutions": overall_word_out.substitutions,
            "deletions": overall_word_out.deletions,
            "insertions": overall_word_out.insertions,
            "hits": overall_word_out.hits
        })

        if lang_refs:
            for l_id in sorted(lang_refs.keys()):
                l_refs = lang_refs[l_id]
                l_hyps = lang_hyps[l_id]
                if l_refs and l_hyps:
                    l_word_out = jiwer.process_words(l_refs, l_hyps)
                    l_char_out = jiwer.process_characters(l_refs, l_hyps)
                    overall_rows.append({
                        "language_id": l_id,
                        "total_samples": len(l_refs),
                        "wer": f"{l_word_out.wer:.4f}",
                        "cer": f"{l_char_out.cer:.4f}",
                        "substitutions": l_word_out.substitutions,
                        "deletions": l_word_out.deletions,
                        "insertions": l_word_out.insertions,
                        "hits": l_word_out.hits
                    })

    summary_fieldnames = [
        "language_id",
        "total_samples",
        "wer",
        "cer",
        "substitutions",
        "deletions",
        "insertions",
        "hits"
    ]

    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(overall_rows)

    print(f"✅ 整體評估結果已儲存至: {results_csv}")

if __name__ == "__main__":
    input_csv = "/mnt/usr1/avi/Projects/whisper_code/transcriptions.csv"
    details_csv = "/mnt/usr1/avi/Projects/whisper_code/eval_details.csv"
    results_csv = "/mnt/usr1/avi/Projects/whisper_code/eval_results.csv"
    evaluate_transcriptions_csv(input_csv, details_csv, results_csv)