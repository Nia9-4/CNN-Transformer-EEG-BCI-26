#!/bin/bash
##SBATCH --job-name="CNN_Transformer_EEG_BCI_26"
##SBATCH --time=12:00:00
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=4
##SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --output=logs/result_%j.out
##SBATCH --error=logs/result_%j.err

##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=areul@uni-osnabrueck.de

# module load cuda/11.8 
set -euo pipefail
source "/home/student/a/areul/miniconda/etc/profile.d/conda.sh"
conda activate dl26
PROJECT_ROOT="/home/student/a/areul/CNN-Transformer-EEG-BCI-26"
cd "$PROJECT_ROOT"

export MPLBACKEND=Agg

python -u src/preprocess.py