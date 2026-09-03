#!/bin/bash
#PBS -N mi_h_sweep_10s_0825
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 1-4

cd $PBS_O_WORKDIR

module load Python/3.11.5-GCCcore-13.2.0
source /rds/general/user/cen25/home/multilayer-neuropil/venv/bin/activate

H_VALUES=(12 14 16 18)

H=${H_VALUES[$PBS_ARRAY_INDEX-1]}

INPUT_INTERNAL_DIR="output/internal/mi_r_ee_small_sweep_2_uniform_seed_3_10s_2.0"
OUTPUT_INTERNAL_DIR="output/internal/mi_h_sweep_10s_0825_${H}"
mkdir -p "$OUTPUT_INTERNAL_DIR"

python network_analysis.py --input-internal-dir "$INPUT_INTERNAL_DIR" \
    --output-internal-dir "$OUTPUT_INTERNAL_DIR" \
    --output-file "output/public/mi_h_sweep_10s_0825_${H}.csv" \
    --mi-bin-width-ms 10 \
    --recompute-mi-from-spikes \
    --mi-lag-ms 0 \
    --summary-signal g_e \
    --summary-dt-ms 0.1 \
    --duration-ms 10000 \
    --r-ee 2.0 \
    --layer-weight-decay-lambda 1.0 \
    --distance-based-mi-h "$H" \
    --distance-based-mi-only