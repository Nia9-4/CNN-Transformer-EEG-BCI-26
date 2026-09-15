#!/bin/bash
#SBATCH --job-name="CNN_Transformer_EEG_BCI_26"
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/result_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=areul@uni-osnabrueck.de

module load cuda/11.8
source activate dl26

export MPLBACKEND=Agg

python -u src/train.py

