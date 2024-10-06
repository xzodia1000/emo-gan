#!/bin/bash -l
 
#SBATCH --job-name=train_gan
#SBATCH --output=job-%j.output
#SBATCH --error=job-%j.error
#SBATCH --time=7-00:00:00
#SBATCH --mem=102400
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu
 
# Change directory to the temporary directory on the compute node
cd /mnt/scratch/users/mna2002/emo-gan/
 
flight start
 
# Activate Gridware
flight env activate conda@gpu
 
python3 train.py -o "./outputs/train_gan/train_6/" -d "dmog" -p "log"
