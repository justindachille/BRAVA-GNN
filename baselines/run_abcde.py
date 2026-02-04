import argparse
import subprocess
import time
import os
import shutil
import re
import sys
import fcntl

# --- MAIN SCRIPT ---

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

experiment_name = "abcde_run"
# Deterministic experiment path based on new ExperimentSetup in util.py
exp_dir = f"experiments/{experiment_name}_{args.seed}"

print(f"--- Running ABCDE | Seed {args.seed} ---")
print(f"Output dir: {exp_dir}")

# Ensure clean start by removing previous run for this seed if it exists
if os.path.exists(exp_dir):
    print(f"Removing existing experiment dir: {exp_dir}")
    shutil.rmtree(exp_dir)

start_time = time.time()

# 1. Run Training
# calls the modified abcde/train.py which now writes to experiments/{experiment_name}_{seed}
train_cmd = ["python", "-u", "abcde/abcde/train.py", "--seed", str(args.seed), "--name", experiment_name]

# Ensure we catch output for debugging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/abcde_train_S{args.seed}.log"

try:
    with open(log_file, "w") as out:
        subprocess.check_call(train_cmd, stdout=out, stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as e:
    print(f"Training failed. Logs in {log_file}")
    sys.exit(1)

training_time = time.time() - start_time
print(f"Training finished in {training_time:.2f}s")

# 2. Find Best Checkpoint
ckpt_dir = os.path.join(exp_dir, "models")
if not os.path.exists(ckpt_dir):
    print(f"Error: Checkpoint directory not found at {ckpt_dir}")
    sys.exit(1)

checkpoints = []
for f in os.listdir(ckpt_dir):
    if f.endswith(".ckpt"):
        checkpoints.append(os.path.join(ckpt_dir, f))

if not checkpoints:
    print(f"Error: No checkpoints generated in {ckpt_dir}")
    sys.exit(1)

# Sort by validation score (val_kendal=XX.XX)
try:
    best_ckpt = sorted(checkpoints, key=lambda x: float(re.search(r"val_kendal=([0-9\.\-]+)", x).group(1)), reverse=True)[0]
except (AttributeError, ValueError):
    print("Warning: Could not parse score. Using newest.")
    best_ckpt = max(checkpoints, key=os.path.getmtime)

print(f"Evaluating checkpoint: {best_ckpt}")

# 3. Run Evaluation
eval_cmd = ["python", "-u", "baselines/eval_abcde.py", "--model_path", best_ckpt, "--algorithm_name", "ABCDE_Train"]
subprocess.check_call(eval_cmd)

# 4. Log Training Time
if not os.path.exists("results"):
    os.makedirs("results")
results_file_training = "results/all_results_training_time.csv"

with open(results_file_training, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, 2)
    if f.tell() == 0:
        f.write("Algorithm,Time\n")
    f.write(f"ABCDE_Train,{training_time:.4f}\n")
    fcntl.flock(f, fcntl.LOCK_UN)

print("Done.")