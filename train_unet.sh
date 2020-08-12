#!/bin/bash
#SBATCH --time=24:0:0 --ntasks=4 --mem=32000M --nodes=1 --gres=gpu:1 --account=def-petersv
module load python/3.6
module load scipy-stack

source ${HOME}/.virtualenvs/CompNet/bin/activate
cd ${HOME}/projects/def-petersv/mattlk/workplace/CompNet/src
python train.py

