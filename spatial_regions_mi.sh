#!/bin/bash
#PBS -N spatial_regions_mi_sweep_seed_4_reduced_null
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=1:mem=32gb

cd $PBS_O_WORKDIR

module load Python/3.11.5-GCCcore-13.2.0
source /rds/general/user/cen25/home/multilayer-neuropil/venv/bin/activate

python regenerate_all_figures.py --radius-um 25.0 \
    --output-dir "spatial_mi_sweep_seed_4_reduced_null" \
    --layer-weight-decay-lambda 1.0 \
    --summary-signal g_e \
    --summary-dt-ms 0.1 \
    --mi-bin-width-ms 10 \
    --duration-ms 10000 \
    --distance-based-mi-h 50 \
    --n-uniform-layers 2 \
    --r-ee 2.0 \
    --seed 4 \
    --spatial-mi-regions 9 16 25 36 49 64 81