from pathlib import Path

import numpy as np
from scipy import sparse


def load_raw_voltage_bundle(npz_path='output/internal/lower_layer_voltage_raw.npz'):
	"""Load raw lower-layer voltages and geometry exported by multilayer.py."""
	bundle = np.load(npz_path)
	return {
		'voltage_lower_exc': np.asarray(bundle['voltage_lower_exc'], dtype=np.float32),
		'time_ms': np.asarray(bundle['time_ms'], dtype=np.float32),
		'lower_layer_indices': np.asarray(bundle['lower_layer_indices'], dtype=np.int32),
		'lower_positions_um': np.asarray(bundle['lower_positions_um'], dtype=np.float32),
		'readout_positions_um': np.asarray(bundle['readout_positions_um'], dtype=np.float32),
	}


def build_radius_neighborhood_matrix(lower_positions_um, readout_positions_um, radius_um):
	"""Create a CSR matrix mapping readout neurons to lower-layer neighbors.

	lower_positions_um shape: (n_layers_lower, n_lower_per_layer, 2)
	readout_positions_um shape: (n_readout, 2)
	returns: CSR matrix shape (n_readout, n_layers_lower * n_lower_per_layer)
	"""
	if radius_um <= 0:
		raise ValueError('radius_um must be positive.')

	lower_flat = np.asarray(lower_positions_um, dtype=np.float32).reshape(-1, 2)
	readout = np.asarray(readout_positions_um, dtype=np.float32)
	radius_sq = float(radius_um) * float(radius_um)

	rows = []
	cols = []
	data = []
	for readout_idx in range(readout.shape[0]):
		dx = lower_flat[:, 0] - readout[readout_idx, 0]
		dy = lower_flat[:, 1] - readout[readout_idx, 1]
		in_radius = np.flatnonzero(dx * dx + dy * dy <= radius_sq)
		if in_radius.size == 0:
			continue

        # TODO: Consider if we want weight to take into account the number of 
		# neighbors, or if it should be independent, or just a function of
        # layer. In the case of uniformly distributed neurons, it shouldn't 
		# matter much.
		weight = np.float32(1.0 / in_radius.size)
		rows.extend([readout_idx] * int(in_radius.size))
		cols.extend(in_radius.tolist())
		data.extend([weight] * int(in_radius.size))

	n_readout = readout.shape[0]
	n_lower_total = lower_flat.shape[0]
	return sparse.csr_matrix((data, (rows, cols)), shape=(n_readout, n_lower_total), dtype=np.float32)


def compute_readout_tensor(voltage_lower_exc, neighborhood_csr):
	"""Compute mean neighborhood activation for each readout neuron at each time.

	voltage_lower_exc shape: (n_layers_lower, n_lower_per_layer, n_time)
	neighborhood_csr shape: (n_readout, n_layers_lower * n_lower_per_layer)
	returns: readout tensor shape (n_time, n_readout)
	"""
	voltages = np.asarray(voltage_lower_exc, dtype=np.float32)
	n_time = voltages.shape[2]

	lower_flat_time = voltages.reshape(-1, n_time)
	readout_by_time = neighborhood_csr @ lower_flat_time
	return np.asarray(readout_by_time.T, dtype=np.float32)


def generate_readout_tensor_from_file(
	radius_um,
	input_npz='output/internal/lower_layer_voltage_raw.npz',
	output_npz='output/internal/readout_avg_radius.npz',
):
	"""Load raw data, build neighborhood matrix, and write readout tensor to disk."""
	bundle = load_raw_voltage_bundle(input_npz)
	neighborhood = build_radius_neighborhood_matrix(
		bundle['lower_positions_um'],
		bundle['readout_positions_um'],
		radius_um=radius_um,
	)
	readout_tensor = compute_readout_tensor(bundle['voltage_lower_exc'], neighborhood)

	Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
	sparse.save_npz('output/internal/readout_neighborhood_radius.npz', neighborhood)
	np.savez_compressed(
		output_npz,
		readout_avg=readout_tensor,
		time_ms=bundle['time_ms'],
		radius_um=np.float32(radius_um),
	)
	return readout_tensor, neighborhood
