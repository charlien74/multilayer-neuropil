# Multilayer Neuropil

Simulation and analysis code for studying how neuronal assembly structure is
transformed across a multilayer spiking network and a spatially averaged
neuropil readout.

The main workflow builds a five-layer Brian2 network, constructs a neuropil
readout from activity in the lower layers, and compares the structural and
functional networks produced by the multilayer and neuropil models.

## Installation

Python 3.11 is recommended. From the repository root:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

The principal dependencies are Brian2, NumPy, Matplotlib, SciPy, Numba, and
NetworkX.

## End-to-End Workflow

Run the complete three-stage pipeline with:

```bash
python regenerate_all_figures.py
```

This command runs, in order:

1. `multilayer.py` to simulate the multilayer network and save its readout
   spikes, summary signals, mutual-information (MI) matrix, and structural
   adjacency matrix.
2. `neuropil.py` to spatially average the saved lower-layer signal, simulate a
   uniform readout layer, and save its spikes and MI matrix.
3. `network_analysis.py` to compare structural and functional networks using
   null-corrected DeltaCon similarity, Louvain partition similarity, and
   distance-based MI.

Figures and tabular results are written to `output/public/`. Reusable arrays,
sparse matrices, and other intermediate data are written to
`output/internal/`. Existing network comparison files are append-only.

The settings used by the PBS launcher are equivalent to:

```bash
python regenerate_all_figures.py \
    --radius-um 10.0 \
    --layer-weight-decay-lambda 1.0 \
    --summary-signal g_e \
    --summary-dt-ms 0.5 \
    --mi-bin-width-ms 10 \
    --duration-ms 2000 \
    --n-uniform-layers 2
```

Use `--output-dir` to isolate a run below both output roots, which is useful
for parameter sweeps and parallel jobs:

```bash
python regenerate_all_figures.py --seed 3 --r-ee 1.5 --output-dir seed_3_r_ee_1_5
```

## Pipeline Parameters

The end-to-end driver exposes the main simulation and analysis settings:

| Option | Default | Description |
| --- | ---: | --- |
| `--duration-ms` | `2000` | Simulation duration in milliseconds. |
| `--seed` | `2` | Random seed propagated through all three stages. |
| `--r-ee` | `2.0` | Within/between-community excitatory connectivity ratio. |
| `--n-uniform-layers` | `2` | Number of assembly-free layers at the top of the stack. |
| `--summary-signal` | `v` | Lower-layer signal used for neuropil construction: `v`, `i_syn`, or `g_e`. |
| `--summary-dt-ms` | `0.1` | Sampling interval for the exported summary signal. |
| `--radius-um` | `25` | Spatial neighborhood radius of each neuropil readout neuron. |
| `--layer-weight-decay-lambda` | `1.0` | Exponential decay rate for weighting source layers by depth. |
| `--mi-bin-width-ms` | `5` | Spike bin width used to construct the pipeline MI matrices. |
| `--mi-lag-ms` | `10` | Lag window used when analysis recomputes MI from spikes. |
| `--louvain-resolution` | `1.0` | Resolution used for Louvain community detection. |
| `--distance-based-mi-h` | `10` | Neighborhood size for distance-based MI. |
| `--distance-based-mi-only` | off | Skip the full network comparison for expedited sweeps. |
| `--spatial-mi-regions` | none | Perfect-square region counts for an optional spatial coarse-graining sweep. |
| `--spatial-mi-bin-width-ms` | `20` | Spike bin width for the spatial MI sweep. |
| `--spatial-mi-radius-um` | `100` | Half-width of the fixed spatial MI grid. |
| `--output-dir` | `.` | Relative run directory below `output/public` and `output/internal`. |

For example, the optional spatial analysis can be run with:

```bash
python regenerate_all_figures.py --spatial-mi-regions 4 9 16 25 36
```

Run `python regenerate_all_figures.py --help` for the authoritative option
list.

## Outputs

Representative public outputs include:

- `spatial_and_raster_all_layers.png`: spatial layouts and spike rasters.
- `spatial_structure_3d_columns.png`: three-dimensional network structure.
- `S_hat_values.txt`: assembly statistic by layer.
- `mi_heatmap_multilayer_readout_exc.png`: multilayer readout MI matrix.
- `neuropil_readout_raster.png`: neuropil readout spike raster.
- `neuropil_proxy_column_mapping.png`: readout-to-proxy-column mapping.
- `mi_heatmap_neuropil_readout_exc.png`: neuropil readout MI matrix.
- `network_compare_results.txt`: append-only network comparison results.
- `spatial_region_network_compare_results.csv`: optional spatially
  coarse-grained comparisons.

Representative internal outputs include:

- `lower_layer_voltage_raw.npz`: sampled lower-layer signal and geometry used
  by the neuropil stage. The historical filename is retained for all choices
  of `--summary-signal`.
- `multilayer_readout_spikes.npz` and `readout_layer_simulation.npz`: saved
  spike events and readout simulation data.
- `mi_matrix_multilayer_readout_exc.npz` and
  `mi_matrix_neuropil_readout_exc.npz`: functional MI matrices.
- `adjacency_global_sparse.npz` and `adjacency_global_layer_offsets.csv`:
  global weighted adjacency data and population offsets.
- `readout_avg_radius.npz` and `readout_neighborhood_radius.npz`: neuropil
  drive and sparse neighborhood mapping.
- `mi_matrix_multilayer_regions_*.npz`: optional region-level MI matrices.

Sweep summaries, archived runs, and thesis figures are kept in the other
subdirectories below `output/`.

## Running Stages Separately

Run only the multilayer simulation and suppress interactive plot windows:

```bash
python multilayer.py --no-show
```

After `multilayer.py` has produced `output/internal/lower_layer_voltage_raw.npz`,
run the neuropil stage with:

```bash
python neuropil.py --radius-um 25 --duration-ms 2000
```

Then compare the saved networks with:

```bash
python network_analysis.py
```

Each script supports `--help`. When running stages independently, keep the
duration, seed, signal metadata, and input/output directories consistent.

## Repository Structure

- `multilayer.py`: Brian2 multilayer simulation, assembly statistics,
  structural exports, readout spikes, and MI outputs.
- `neuropil.py`: spatial neuropil aggregation and uniform readout-layer
  simulation.
- `network_analysis.py`: structural/functional comparison, Louvain metrics,
  null-corrected DeltaCon, distance-based MI, and spatial coarse-graining.
- `regenerate_all_figures.py`: reproducible end-to-end pipeline driver.
- `model_util.py`: shared model equations, parameters, geometry, connectivity,
  and assembly-statistic utilities.
- `mutual_information.py`: reusable spike-train MI calculations and outputs.
- `spectral_analysis.py`: adjacency extraction and eigenspectrum utilities.
- `generate_s_figure.py`: post-processing for assembly-statistic sweep plots.
- `one_d.py`: one-dimensional simulation variant.
- `analysis.ipynb`: exploratory analysis and thesis figure generation.
- `run_neuropil.sh`: PBS batch launcher for the end-to-end workflow.
- `bin_sweep.sh`: PBS array job for sweeping MI spike-bin widths.
- `louvain_sweep.sh`: PBS array job for sweeping Louvain resolutions.
- `network_analysis_sweep.sh`: PBS array job for sweeping the neighborhood
  size used by distance-based MI analysis.
- `spatial_regions_mi.sh`: PBS launcher for an end-to-end spatial-region MI
  sweep.
- `util/spike_mi.py`: supporting spike MI implementation.

## Batch Execution

The shell launchers are configured for a PBS environment. They expect
`$PBS_O_WORKDIR`, the Imperial RCS Python module, and the virtual environment
paths specified in the scripts. Submit the main workflow with:

```bash
qsub run_neuropil.sh
```

The random seed can be overridden for a batch job:

```bash
qsub -v SEED=4 run_neuropil.sh
```

The analysis sweeps can be submitted with:

```bash
qsub bin_sweep.sh
qsub louvain_sweep.sh
qsub network_analysis_sweep.sh
qsub spatial_regions_mi.sh
```

These scripts contain experiment-specific input directories, parameter ranges,
and output names; update those values in the scripts when targeting another
saved run.

Model-scale constants and equations are defined in `model_util.py`; prefer the
command-line options above for reproducible experimental changes.
