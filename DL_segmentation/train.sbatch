#!/bin/bash 
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --account=pr_211_general
#SBATCH --job-name=train
#SBATCH --output=train.out


singularity \
    exec  --nv \
    --overlay /scratch/mz2466/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
python3 train.py --config_path='baseline.deeplabv3p' \
                 --model_dir='./log/normal_baseline/deeplabv3p_bs16' \
                 train.eval_interval_epoch 1
"