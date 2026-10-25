# Whisper Smoke Test & Loss Visualization Guide

This guide contains:
1. **Ready-to-run Smoke Test Commands** (Quick 5-step test & 60-step test).
2. **How Hugging Face Trainer Stores Loss Values** (`trainer_state.json`).
3. **Python Script to Extract Loss & Generate Loss Curve Images**.
4. **How to Enable TensorBoard / Weights & Biases (WandB) for Real-Time Diagrams**.

---

## 1. Manual Smoke Test Commands

Activate your conda environment first:
```bash
conda activate whisper_ft
cd /mnt/usr1/avi/Projects/formosan/SARC_Rookie_Code
```

### Option A: Ultra-Fast Smoke Test (5 steps, ~10 seconds)
Use this to quickly verify code syntax, token embedding resizing, and checkpoint saving:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name openai/whisper-tiny \
  --language "<|formosan|>" \
  --max_train_samples 32 \
  --max_eval_samples 16 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 4 \
  --max_steps 5 \
  --eval_steps 5 \
  --save_steps 5 \
  --logging_steps 1 \
  --num_proc 2 \
  --output_dir /mnt/usr1/avi/Projects/whisper_code/whisper-smoke-test-5steps
```

---

### Option B: Medium Smoke Test (60 steps, ~35 seconds)
Use this to see clear loss convergence, validation improvement, and run real inference:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name openai/whisper-tiny \
  --language "<|formosan|>" \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 8 \
  --learning_rate 1e-4 \
  --warmup_steps 10 \
  --max_steps 60 \
  --eval_steps 15 \
  --save_steps 15 \
  --logging_steps 5 \
  --num_proc 4 \
  --output_dir /mnt/usr1/avi/Projects/whisper_code/whisper-smoke-test-60steps
```

---

### Run Inference with the Checkpoint
Test any `.wav` audio file using your trained model:

```bash
python inference.py \
  --model_path /mnt/usr1/avi/Projects/whisper_code/whisper-smoke-test-60steps/final_model \
  --audio_path /nfs/DGX2/Workspace/avi/FormosanBank/Corpora/ePark/Audio/sheng_huo_hui_hua_pian_daily_conversation/Amis/Coastal_Amis/sheng_huo_hui_hua_pian_daily_conversation_Coastal_Amis_0.wav \
  --language "<|formosan|>"
```

---

## 2. Where Does Hugging Face Store Loss Values?

During training, Hugging Face `Trainer` logs loss at every `logging_steps` and `eval_steps`. These numbers are automatically saved inside:

```
<output_dir>/checkpoint-<step>/trainer_state.json
```
*(or `<output_dir>/trainer_state.json` if training finishes or state is saved)*

### Structure of `trainer_state.json`
Inside the JSON file, the key `"log_history"` is a list of log entries:
```json
{
  "log_history": [
    {
      "step": 5,
      "loss": 10.9588,
      "learning_rate": 4e-05,
      "epoch": 0.08
    },
    {
      "step": 15,
      "eval_loss": 1.3207,
      "epoch": 0.24
    }
  ]
}
```

---

## 3. Python Script to Plot the Loss Curve

You can run `plot_loss.py` with `--model_path` pointing directly to any model path:
- A `final_model` folder
- A `checkpoint-<step>` folder
- The root training directory

```bash
# 1. Point to final_model
python plot_loss.py --model_path /mnt/usr1/avi/Projects/whisper_code/whisper-smoke-test-60steps/final_model

# 2. Point to a specific checkpoint
python plot_loss.py --model_path /mnt/usr1/avi/Projects/whisper_code/whisper-smoke-test-60steps/checkpoint-60

# 3. Custom output file name
python plot_loss.py --model_path /mnt/usr1/avi/Projects/whisper_code/whisper-smoke-test-60steps/final_model --save_path my_loss.png
```

---

## 4. Real-Time Tracking: TensorBoard & WandB

If you prefer seeing live graphs in your browser while the model is training:

### A. Using TensorBoard
1. In `train.py`, change:
   ```python
   report_to="tensorboard", # instead of "none"
   logging_dir=f"{args.output_dir}/runs",
   ```
2. Launch TensorBoard in your terminal:
   ```bash
   tensorboard --logdir /mnt/usr1/avi/Projects/whisper_code/whisper-smoke-test-60steps/runs --port 6006
   ```
3. Open `http://localhost:6006` in your browser to view real-time loss, learning rate, and evaluation curves.

### B. Using Weights & Biases (WandB)
1. Install and login:
   ```bash
   pip install wandb
   wandb login
   ```
2. In `train.py`, set `report_to="wandb"`. Hugging Face will automatically create a dashboard with interactive loss charts for each experiment.
