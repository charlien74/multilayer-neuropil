#!/bin/bash
#PBS -N neuropil
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=1:mem=16gb

cd $PBS_O_WORKDIR

module load Python/3.11.5-GCCcore-13.2.0
source /rds/general/user/cen25/home/multilayer-neuropil/venv/bin/activate

python regenerate_all_figures.py --radius-um 10.0 --layer-weight-decay-lambda 1.0 --summary-signal g_e --summary-dt-ms 0.1 --mi-bin-width-ms 30 --duration-ms 20000