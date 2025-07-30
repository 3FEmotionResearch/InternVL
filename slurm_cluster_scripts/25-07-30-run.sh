#!/bin/bash
#SBATCH --job-name=25-07-30-emotion_attack_my
#SBATCH --account=project_2011211
#SBATCH --partition=gpusmall
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

export HF_HOME="/scratch/project_2011211/3f_emotion_re/.hf_cache"
cd /scratch/project_2011211/3f_emotion_re/ReposPublic/InternVL

# Clean the environment and load the same module used to create the venv
module purge
module load pytorch/2.3

# Activate virtual environment
source /scratch/project_2011211/3f_emotion_re/py_envs/25-07-29-torch2_3_internvl/bin/activate

# Run your Python script
# srun is used to launch the parallel task
srun python internvl_chat/emotion_attack_my.py
