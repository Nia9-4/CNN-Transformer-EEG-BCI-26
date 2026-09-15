#!/bin/bash
#SBATCH --job-name="CNN_Transformer_EEG_BCI_26"
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/result_j%.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=areul@uos.de

wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x84_64.sh -b -p "miniconda" -u
source miniconda/bin/activate
conda create -y -q --name dl26 python=3.10 -- channel defaults
source activate dl26
pip install --no-cache-dir --upgrade -r requirements.txt

srun python src/train.py