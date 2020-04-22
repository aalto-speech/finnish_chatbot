#!/bin/bash -l

#SBATCH --partition dgx-spa,dgx-common
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/work/jpleino1/log/s2s_chatbot-%j.log
#SBATCH -e /scratch/work/jpleino1/log/s2s_chatbot-%j.log

export=PATH,HOME,USER,TERM,WRKDIR
export PYTHONUNBUFFERED=1

module purge
module load anaconda3
source activate /scratch/work/jpleino1/conda/envs/nmt_chatbot
#conda activate nmt_chatbot
script=$1
cat encoderDecoder_global_variables.py
cat encoderDecoder_prep_data.py
cat encoderDecoder_voc.py

python $script
