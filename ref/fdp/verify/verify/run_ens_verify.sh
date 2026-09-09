#!/bin/bash

#SBATCH -J W2S
#SBATCH --comment=AI
#SBATCH --wckey=104-06
#SBATCH -p serial
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=6144
#SBATCH --gres=gpu:A800:1
#SBATCH --time=120:00:00

. /etc/profile
. ~/.bash_profile

#module load miniconda3/py310_24.5.0-0-Linux-x86_64
module load anaconda/2024.10.23
source activate /gpu/zhaochy/conda3/envs/w2s

BASE=/gpu/zhaochy/fdp2
cd ${BASE}
OUTPUT_BASE=${BASE}/VERIFY
INPUT_BASE=/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026
MODEL_NAME=W2S

today=20260822
#today=`date '+%Y%m%d'`
#TIMESTAMP=${today}00

python ${BASE}/scripts/ensemble_verifier.py --date 2026082200 --cra-root /gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026 --fcstdata-root /gpu/zhaochy/fdp2/FCSTDATA --models Fengqing PuYun YJ-TianJi NJU-Earth W2S
