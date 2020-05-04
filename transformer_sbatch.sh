#!/bin/bash -l

#SBATCH --partition dgx-spa,dgx-common
#SBATCH --time=60:30:00
#SBATCH --mem=32G
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

git log | head -n 6

echo "transformer_main.py"
cat transformer_main.py

echo "transformer_hyperparameters.py"
cat transformer_hyperparameters.py
cat transformer_hyperparameters.py > "$SLURM_JOBID"_transformer_hyperparameters.py

echo "transformer_global_variables.py"
cat transformer_global_variables.py

echo "transformer_prep_data.py"
cat transformer_prep_data.py

echo "transformer_voc.py"
cat transformer_voc.py

echo "transformer_training.py"
cat transformer_training.py

echo "transformer_models.py"
cat transformer_models.py

echo "transformer_sbatch.sh"
cat transformer_sbatch.sh

echo "transformer_evaluation.py"
cat transformer_evaluation.py

echo "transformer_evaluate_script.py"
cat transformer_evaluate_script.py

python $script $SLURM_JOBID
