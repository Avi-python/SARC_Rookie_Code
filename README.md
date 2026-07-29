# NYCU ML2025 HW3 - Taiwanese Speech Recognition

本專案為 Machine Learning 2025 HW3 作業，主題為台語語音辨識 (Taiwanese Speech Recognition)。
主要使用 **OpenAI Whisper Large V3 Turbo** 模型進行 Fine-tuning，並透過 **音訊串接 (Audio Concatenation)** 策略解決短音檔造成的幻覺問題 (Hallucination)。

## 環境需求

- Python (≥ 3.9)
- datasets
- jiCER (用於 CER 計算)
- PyTorch (支援CUDA)
- Transformers
- Torchaudio
- Audiomentations, Nlpaug (資料增強)
- Jiwer (計算MER/WER)
- Pydub (音頻處理)


## 專案架構與檔案說明

本專案主要分為「訓練」、「推論/產生提交檔」與「本地驗證 (MER計算)」三個部分。

### 1. 核心訓練

- **`train.py`**
    - **功能**：模型訓練主程式。
    - **模型**：使用 `openai/whisper-large-v3-turbo`，設定為中文 (`zh`) 聽寫任務。
    - **資料增強 (Data Augmentation)**：
        - **Stage 1 (頻譜層級)**：使用 `nlpaug` 進行 VTLP (Vocal Tract Length Perturbation)。
        - **Stage 2 (波形層級)**：使用 `audiomentations`，包含 GaussianSNR、TimeStretch (變速)、PitchShift (變調)。
        - **環境噪音**：加入 Background Noise (MUSAN)、Short Noises (ESC-50) 與 RIR (SLR28) 模擬真實環境。
    - **流程**：讀取 JSONL 資料 -> 預處理 (Resample 16k) -> 增強 -> Tokenize -> Trainer 訓練 -> 儲存 Checkpoint。

- **`connect_process.py`**
    - **功能**：**音檔串接策略 (Concatenation)**，用於改善 Whisper 對短音檔易產生幻覺的問題。
    - **運作邏輯**：
        1. **動態長度**：將短音檔串接成 10~30 秒的長音檔（符合 Whisper 30s Window 機制）。
        2. **自然停頓**：在音檔間隨機插入 150ms ~ 800ms 的靜音，模擬人類語句間隔。
        3. **標籤同步**：將對應的 Text 使用空格串接。
        4. **優先級**：優先挑選未被使用過的音檔 (Usage Count = 0)，確保資料覆蓋率。
    - **輸出**：產生新的 `.wav` 檔案與對應的 `combined.jsonl` 供 `train.py` 使用。

### 2. 推論與提交 (Kaggle)

- **`result.py`**
    - **功能**：產生 Kaggle 評分用的 Submission CSV。
    - **流程**：
        1. 載入訓練好的 Checkpoint (`checkpoint-xxxx`)。
        2. 遍歷 Test 資料夾內的 `.wav` 檔案。
        3. 強制使用 `language="zh"` 進行 decode。
        4. **後處理**：正規化輸出文字（移除標點符號、保留漢字與空格），並寫入 CSV。

### 3. 本地驗證與 MER 計算流程

這一組程式碼用於在本地端從訓練集 (Train Set) 切分出部分資料進行驗證，計算 Mixed Error Rate (MER) 以評估模型表現。

- **`006_inference.py`**
    - **功能**：模型推論測試。
    - **細節**：從訓練集中抓取前 200 筆音檔進行推論，將結果輸出為 `transcriptions_200.txt` (格式：`檔名 \t 預測文字`)。

- **`007_transcriptions_normalize.py`**
    - **功能**：預測文字正規化。
    - **細節**：讀取 `transcriptions_200.txt`，移除標點符號 (`，。！？` 等) 並處理漢字間距，輸出 `hyp.txt` (Hypothesis)。

- **`008_ref.py`**
    - **功能**：準備標準答案 (Ground Truth)。
    - **細節**：讀取原始 `train.jsonl`，提取對應的真實文本並進行相同的正規化處理，輸出 `ref.txt` (Reference)。

- **`009_mer.py`**
    - **功能**：計算 MER (Mixed Error Rate) 與詳細錯誤分析。
    - **細節**：
        1. 比對 `ref.txt` 與 `hyp.txt`。
        2. 使用 `jiwer` 計算 WER/MER。
        3. 輸出詳細報告：包含 **Deletion (漏字)**、**Insertion (贅字/幻覺)**、**Substitution (錯字)** 的統計與差異標示。

## 實驗筆記

### 為什麼需要 `connect_process.py`?
在 Baseline 實驗中發現，Whisper 模型對於極短的音檔容易產生嚴重的 **Insertion Error (幻覺)**，導致 Insertion Rate 高於 Deletion Rate 數倍。
- **原因**：Whisper 固定輸入 30 秒，短音檔會導致大量 Padding，自回歸解碼器容易在靜音區段「腦補」內容。
- **解法**：透過串接將輸入填滿至 10-30 秒，提供穩定的聲學特徵，顯著降低幻覺產生，提升整體辨識率。

## 使用方式

**1. 資料準備 (串接)**
```python connect_process.py```

**2. 開始訓練**
```python train.py```

**3. 產生 Kaggle 提交檔**
```python result.py```

**4. 本地驗證 (Optional)**
```python 006_inference.py```
```python 007_transcriptions_normalize.py```
```python 008_ref.py```
```python 009_mer.py```

## 討論與心得

這次作業最大的收穫在於深入理解了 Whisper 模型的特性與弱點，並透過資料工程 (Data Engineering) 的手段有效解決問題。

一開始跑 Baseline 時，雖然使用了參數量龐大的 Whisper Large V3 Turbo，但發現對於極短的台語音檔（例如只有 2-3 秒的句子），模型非常容易產生「幻覺」(Hallucination)。具體表現是 Insertion Rate 異常的高，模型常在無聲段落自己生成重複的字詞或無意義的語句。經過分析，確認這是因為 Whisper 架構強制將輸入 Padding 到 30 秒所致；當有效語音過短，剩下的長段空白讓 Autoregressive Decoder 在缺乏聲學資訊的情況下容易失控。

為了處理這個問題，我實作了 connect_process.py 將短音檔動態串接。這不僅僅是為了增加訓練效率，更核心的目的是「填滿 Attention Window」，強迫模型專注於真實的語音訊號。實作細節上，我特別注意了「隨機靜音」的插入（150ms-800ms）。如果直接硬接，模型會學不到語句邊界；加入適當靜音後，模型能更好地區分不同語句。此外，為了避免隨機取樣導致某些檔案沒被練到，我還加入了優先隊列邏輯，優先挑選 usage_count=0 的檔案。

結果非常直觀：Insertion Rate 從 Baseline 的 1% 降到 0.1%，Kaggle Private Score 也從 7.23 大幅進步到 3.04。這證明了在大家都在拼模型參數量時，對資料特性的理解與前處理策略，往往能帶來更具體且顯著的效能提升。
