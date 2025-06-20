#!/bin/bash

#SBATCH -p gpu24
#SBATCH -t 0-3:00:00
#SBATCH -o /CT/HOIMOCAP/work/exps/slurm_outputs/%A_%a.out
#SBATCH -e /CT/HOIMOCAP/work/exps/slurm_outputs/%A_%a.err
#SBATCH -a 1
#SBATCH --gres gpu:1

echo $PWD
source ~/activate_conda_jianchun.sh
eval "$(conda shell.bash hook)"
echo "Activate dut environment"
conda activate dut

config="./configs/s724.yaml"
texResGeo="256"
expName="exp30_texResGeo${texResGeo}_iso0.5_worldLap"
echo "python train_geometry.py --lrDecayStep 1000000 --expName ${expName} \
        --weightLap 1.0 --weightIso 0.5 --weightNmlCons 0.001 --config ${config} --texResGeo ${texResGeo} --noEval --worldLap"
python train_geometry.py --lrDecayStep 1000000 \
        --expName ${expName} \
        --weightLap 1.0 --weightIso 0.5 --weightNmlCons 0.001 \
        --config ${config} \
        --texResGeo ${texResGeo} --noEval --worldLap
