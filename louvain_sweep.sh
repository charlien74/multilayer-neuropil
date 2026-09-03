#!/bin/bash
#PBS -N neuropil_louvain_sweep
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -J 1-8

cd $PBS_O_WORKDIR

module load Python/3.11.5-GCCcore-13.2.0
source /rds/general/user/cen25/home/multilayer-neuropil/venv/bin/activate

RESOLUTION_VALUES=(4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0)

RESOLUTION=${RESOLUTION_VALUES[$PBS_ARRAY_INDEX-1]}

INPUT_INTERNAL_DIR="output/internal"
OUTPUT_INTERNAL_DIR="output/internal/louvain_sweep_${RESOLUTION}"
mkdir -p "$OUTPUT_INTERNAL_DIR"

python network_analysis.py --input-internal-dir "$INPUT_INTERNAL_DIR" \
    --output-internal-dir "$OUTPUT_INTERNAL_DIR" \
    --output-file "output/public/louvain_sweep_${RESOLUTION}.csv" \
    --mi-bin-width-ms 10 \
    --recompute-mi-from-spikes \
    --mi-lag-ms 20 \
    --summary-signal g_e \
    --summary-dt-ms 0.1 \
    --duration-ms 20000 \
    --r-ee 2.0 \
    --layer-weight-decay-lambda 1.0 \
    --louvain-resolution "$RESOLUTION"