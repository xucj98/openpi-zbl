#!/bin/bash
#SBATCH --job-name=xiaoqihezuo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=124
#SBATCH --gres=gpu:8
#SBATCH --mem=1432G
#SBATCH --exclude=master
#SBATCH -o /x2robot/xinyuanfang/small_project/openpi/%x-%j.log

export WANDB_PROXY="http://10.7.145.219:3128"
export HTTPS_PROXY="http://10.7.145.219:3128"
export HTTP_PROXY="http://10.7.145.219:3128"

echo "Starting debug run with minimal resources..."
echo "Available memory: $(free -h)"
echo "Available GPUs: $(nvidia-smi -L)"

# Run the debug training
uv run scripts/train.py test_case

echo "Debug run completed." 