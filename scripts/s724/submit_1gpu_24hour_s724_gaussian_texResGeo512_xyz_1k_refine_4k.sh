#!/bin/bash

#SBATCH -p gpu24
#SBATCH -t 0-23:30:00
#SBATCH -o /CT/HOIMOCAP/work/exps/slurm_outputs/%A_%a.out
#SBATCH -e /CT/HOIMOCAP/work/exps/slurm_outputs/%A_%a.err
#SBATCH -a 1
#SBATCH --gres gpu:1

# setup the slurm
#. ./slurmSetup.sh

echo $PWD
source ~/activate_conda_jianchun.sh
eval "$(conda shell.bash hook)"
echo "Activate dut environment"
conda activate dut

config="./configs/s724.yaml"
texResGau="512"
weightReg="0.01"
deltaType="xyz"
imgScale="1.0"
postScale="1"
expName="exp30_4k_postScale${postScale}_decay650k_texResGau${texResGau}"
ckpt="/CT/HOIMOCAP5/static00/results/DUG/s724_gaussian_exp30_1k_postScale1_decay650k_texResGau512_xyz_0603_212305/ckpt/s724_680000.pth"
echo "python ./train_s3_gaussian_multilevel_new.py --weightColor 1.0 --weightSSIM 0.1 --weightMRF 0.01 --weightReg ${weightReg} \
        --numStep 2000000 --expName ${expName} \
        --withPostScale --imgScale ${imgScale} --deltaType ${deltaType} --lrDecayStep 1900000 --lossFreq2 50000 \
        --withStepDecay --split train --config ${config} --texResGau ${texResGau} --ckpt ${ckpt}"

python ./train_gaussian.py --weightColor 1.0 --weightSSIM 0.1 --weightMRF 0.01 --weightReg ${weightReg} \
        --numStep 2000000 --expName ${expName} \
        --withPostScale --imgScale ${imgScale} --deltaType ${deltaType} --lrDecayStep 1900000 \
        --withStepDecay --split train --config ${config} --texResGau ${texResGau} --lossFreq2 50000 --ckpt ${ckpt}

