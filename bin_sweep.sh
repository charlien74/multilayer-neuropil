#!/bin/bash
#PBS -N bin_sweep
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -J 1-20

cd $PBS_O_WORKDIR

module load Python/3.11.5-GCCcore-13.2.0
source /rds/general/user/cen25/home/multilayer-neuropil/venv/bin/activate

BIN_VALUES=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)

BIN=${BIN_VALUES[$PBS_ARRAY_INDEX-1]}

INPUT_INTERNAL_DIR="output/internal"
OUTPUT_INTERNAL_DIR="output/internal/mi_bin_${BIN}"
mkdir -p "$OUTPUT_INTERNAL_DIR"

python network_analysis.py --input-internal-dir "$INPUT_INTERNAL_DIR" \
    --output-internal-dir "$OUTPUT_INTERNAL_DIR" \
    --output-file "output/public/mi_bin_${BIN}.csv" \
    --mi-bin-width-ms "$BIN" \
    --recompute-mi-from-spikes \
    --mi-lag-ms 20 \
    --summary-signal g_e \
    --summary-dt-ms 0.1 \
    --duration-ms 20000 \
    --r-ee 2.0 \
    --layer-weight-decay-lambda 1.0