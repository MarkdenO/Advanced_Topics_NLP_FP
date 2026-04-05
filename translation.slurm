#!/bin/bash
#SBATCH --job-name=panda_translation
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=8GB
#SBATCH --output=translation.%j.out
#SBATCH --error=translation.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "=== Job Info ==="
echo "Job: $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname)"

module --force purge

# ── Fill these in after running: module avail 2>&1 | grep -iE "cuda|pytorch" ──
module --ignore_cache load CUDA/12.6.0         # <-- replace with exact name from module avail
module --ignore_cache load Python/3.10.8-GCCcore-12.2.0  # <-- replace if needed
# ─────────────────────────────────────────────────────────────────────────────

echo "Loaded modules:"
module list

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Verify GPU is actually visible
nvidia-smi || echo "WARNING: nvidia-smi failed — GPU may not be allocated"

# Activate your venv (must have been created with the same Python/CUDA above)
source /scratch/s4957075/Advanced_Topics_NLP_FP/.venv2/bin/activate

echo "Python: $(which python3)"

# Quick sanity check before running the real script
python3 - <<'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

python3 translate_panda_dataset.py -n 50000 -b 256