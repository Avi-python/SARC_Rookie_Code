#!/usr/bin/env python3
"""
Plot Training & Evaluation Loss Curve from Hugging Face Trainer State
Usage:
    python plot_loss.py --model_path /mnt/usr1/avi/Projects/whisper_code/whisper-formosan-test-run/final_model
    python plot_loss.py --model_path /mnt/usr1/avi/Projects/whisper_code/whisper-formosan-test-run/checkpoint-60
    python plot_loss.py --model_path /mnt/usr1/avi/Projects/whisper_code/whisper-formosan-test-run
"""

import os
import glob
import json
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot loss curves from trainer_state.json")
    parser.add_argument(
        "--model_path",
        "--output_dir",
        dest="model_path",
        type=str,
        default="/mnt/usr1/avi/Projects/whisper_code/whisper-formosan-test-run",
        help="Path to model directory, checkpoint folder, or training output directory containing trainer_state.json",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output image path (default: <model_dir>/loss_curve.png)",
    )
    return parser.parse_args()


def resolve_state_file(path):
    """
    Find trainer_state.json from path, handling:
    1. Direct path to trainer_state.json
    2. Path to checkpoint folder (e.g. checkpoint-60)
    3. Path to final_model folder (checks parent run directory)
    4. Path to root training output directory
    Returns:
        (state_file_path, run_dir_path)
    """
    if os.path.isfile(path):
        if path.endswith(".json"):
            run_dir = os.path.dirname(os.path.abspath(path))
            return path, run_dir
        path = os.path.dirname(os.path.abspath(path))

    abs_path = os.path.abspath(path)

    # 1. If path is a subfolder (e.g., 'final_model' or 'checkpoint-xxx'), find its parent run directory
    basename = os.path.basename(abs_path)
    if basename == "final_model" or basename.startswith("checkpoint-"):
        run_dir = os.path.dirname(abs_path)
    else:
        run_dir = abs_path

    # Priority A: <run_dir>/trainer_state.json (official Trainer.save_state() location with complete history)
    run_state = os.path.join(run_dir, "trainer_state.json")
    if os.path.isfile(run_state):
        return run_state, run_dir

    # Priority B: inside the requested directory itself (e.g. if pointing to a checkpoint directly)
    direct_state = os.path.join(abs_path, "trainer_state.json")
    if os.path.isfile(direct_state):
        return direct_state, run_dir

    # Priority C: newest checkpoint under run_dir
    checkpoint_states = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    if checkpoint_states:
        return sorted(checkpoint_states, key=os.path.getmtime)[-1], run_dir

    # Priority D: recursive search
    recursive_states = glob.glob(os.path.join(run_dir, "**", "trainer_state.json"), recursive=True)
    if recursive_states:
        return sorted(recursive_states, key=os.path.getmtime)[-1], run_dir

    return None, run_dir


def main():
    args = parse_args()
    model_path = os.path.abspath(args.model_path)
    state_file, run_dir = resolve_state_file(model_path)

    if not state_file:
        print(f"[ERROR] Could not find trainer_state.json for path '{args.model_path}'")
        return

    print(f"[INFO] Reading loss history from: {state_file}")

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    if not train_steps and not eval_steps:
        print("[WARNING] No loss entries found in log_history.")
        return

    plt.figure(figsize=(9, 5))
    if train_steps:
        plt.plot(train_steps, train_losses, label="Training Loss", color="#1f77b4", marker="o", linewidth=2)
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="#d62728", marker="s", linewidth=2, linestyle="--")

    plt.title("Whisper Fine-Tuning Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Step", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Save to run_dir (keeps final_model/ clean for deployment) unless save_path is explicitly passed
    save_path = args.save_path or os.path.join(run_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Loss curve plot saved to: {save_path}")


if __name__ == "__main__":
    main()
