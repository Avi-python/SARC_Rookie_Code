#!/usr/bin/env python3
"""
clean_asr_text.py

A standalone text normalization script for Formosan language ASR inference and evaluation.
Strips non-semantic grammatical punctuation and excess whitespace while preserving
language-specific phonemes (e.g., glottal stops, colons, circumflexes, internal hyphens).
"""

import argparse
import re
import sys

# Non-phonemic sentence punctuation to strip during text cleaning.
# IMPORTANT: Explicitly preserves:
#   ^ (circumflex / caret: phonemic consonant in Amis)
#   : (colon: phonemic vowel length/distinction)
#   ' (straight apostrophe: glottal stops across Formosan languages)
#   - (hyphens when internal to words/compounds)
PUNCT_STRIP_RE = re.compile(r'[\.,\?!;;"“”…\(\)\[\]\{\}\<\>_`~—/\\，。！？；：“”（）《》【】、]')
WHITESPACE_RE = re.compile(r'\s+')

def normalize_asr_text(text: str, lowercase: bool = True) -> str:
    """
    Normalize text for ASR evaluation or inference.
    """
    if not text:
        return ""
        
    # 1. Normalize typographic / curly apostrophes to straight ASCII single quotes
    #    so they are correctly preserved as glottal stops.
    cleaned = text.replace("’", "'").replace("‘", "'") # This is 英文撇號

    # Kaldi special rule
    cleaned = cleaned.replace("ʼ", "'") # This is 修飾字母撇號
    cleaned = cleaned.replace("⌃", "^") # This two is different

    # 2. Strip non-phonemic punctuation (replaced with space to prevent word fusion)
    cleaned = PUNCT_STRIP_RE.sub(" ", cleaned)
    
    # 3. Strip leading, trailing, or isolated hyphens (preserves internal hyphens)
    cleaned = re.sub(r'(^|\s)-+|-+(\s|$)', ' ', cleaned)
    
    # 4. Normalize and collapse whitespace
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    
    # 5. Case folding (essential for WER evaluation)
    if lowercase:
        cleaned = cleaned.lower()
        
    return cleaned

def main():
    parser = argparse.ArgumentParser(
        description="Normalize Formosan ASR inference text for WER evaluation."
    )
    parser.add_argument(
        "input_file", nargs="?", type=argparse.FileType("r"), default=sys.stdin,
        help="Input text file (reads from stdin if not provided)."
    )
    parser.add_argument(
        "--preserve-case", action="store_true",
        help="Do not lowercase the text."
    )
    args = parser.parse_args()

    for line in args.input_file:
        cleaned_line = clean_asr_text(line, lowercase=not args.preserve_case)
        print(cleaned_line)

if __name__ == "__main__":
    main()
