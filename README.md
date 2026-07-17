# multilayer-neuropil
Using a neuronal model consisting of multiple layers to explore neuropil dynamics.

## Repository Structure

- `multilayer.py`
	- Main simulation entry point.
	- Builds a 5-layer network in Brian2 with excitatory and inhibitory populations per layer.
	- Uses distance-dependent intra-layer connectivity and distance-decaying inter-layer connectivity.
	- Runs the simulation, prints spike counts and assembly metric values (`S`, `<S_shuff>`, `S - <S_shuff>`), and saves a summary figure.

- `model_util.py`
	- Shared constants and model definitions (timing, neuron counts, membrane/synapse equations).
	- Layout and connectivity helpers for clustered pentacle geometry and uniform layouts.
	- Utility functions for assembly statistics (including shuffled controls).

- `requirements.txt`
	- Python dependencies required to run the simulation.

- `output/`
	- Directory where generated plots are written.

## Running `multilayer.py`

1. Create/activate a Python environment (recommended).
2. Install dependencies:

	 ```bash
	 pip install -r requirements.txt
	 ```

3. Run the simulation from the repository root:

	 ```bash
	 python multilayer.py
	 ```

### What the script does at runtime

- Initializes model parameters and random seed.
- Creates spatial neuron layouts for each layer (top layer uses a uniform layout).
- Constructs intra-layer and inter-layer synapses.
- Runs the network for the configured duration.
- Prints per-layer spike counts and assembly metrics to stdout.
- Saves the combined spatial/raster figure to:
	- `output/spatial_and_raster_all_layers.png`
	- `output/spatial_structure_3d_columns.png` (3D 5-layer structural view with sampled dense intra-column E-E edges)
	- `output/adjacency_global_sparse.npz` (global weighted adjacency from Brian2 synapses in CSR arrays)
	- `output/adjacency_global_layer_offsets.csv` (row/column offsets for each layer's E and I blocks)
	- `output/eigenvalues_global_dominant.png` (dominant global eigenvalues in the complex plane)
	- `output/eigenvalues_layer_ee.png` (full E-E eigenspectrum per layer)

## Common Parameters to Modify

If you want to experiment with behavior, these are good starting points:

- In `multilayer.py`:
	- `N_layers`
	- `TARGET_AVG_P_OVERALL`
	- `max_p_exc_interlayer`, `max_p_inh_interlayer`
	- `interlayer_decay_l`

- In `model_util.py`:
	- `duration`, `defaultclock.dt`
	- `N_exc_c`, `N_inh`
	- geometry terms such as `R`, `sigma_c`, `sigma_connection`
