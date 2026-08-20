import argparse
from pathlib import Path

import numpy as np
from brian2 import *
from scipy import sparse

from model_util import *
from mutual_information import save_mi_outputs


w_ee_base = 4 * 0.0156 * kHz
w_ei_base = -0.0297 * kHz
w_ie_base = 0.0074 * kHz
w_ii_base = -0.0297 * kHz

p_avg = 0.02
weight_decay_l = 50 * um
inh_weight_decay_l = 100 * um

eqs_exc_readout = """
dv/dt = (mu - v) / tau_m + g_e + g_i + I_ext : 1 (unless refractory)
dg_e/dt = -g_e / tau_e : Hz
dg_i/dt = -g_i / tau_i : Hz
I_ext = readout_drive(t, i) * Hz : Hz
mu : 1
tau_m : second (constant)
column_id : integer (constant)
x: meter
y: meter
"""

eqs_inh_readout = """
dv/dt = (mu - v) / tau_m + g_e + g_i : 1 (unless refractory)
dg_e/dt = -g_e / tau_e : Hz
dg_i/dt = -g_i / tau_i : Hz
mu : 1
tau_m : second (constant)
column_id : integer (constant)
x: meter
y: meter
"""


def load_raw_voltage_bundle(npz_path='output/internal/lower_layer_voltage_raw.npz'):
	"""Load raw lower-layer voltages and geometry exported by multilayer.py."""
	with np.load(npz_path) as bundle:
		return {
			'voltage_lower_exc': np.asarray(bundle['voltage_lower_exc'], dtype=np.float32),
			'time_ms': np.asarray(bundle['time_ms'], dtype=np.float32),
			'lower_layer_indices': np.asarray(bundle['lower_layer_indices'], dtype=np.int32),
			'lower_positions_um': np.asarray(bundle['lower_positions_um'], dtype=np.float32),
			'readout_positions_um': np.asarray(bundle['readout_positions_um'], dtype=np.float32),
		}


def build_radius_neighborhood_matrix(
	lower_positions_um,
	readout_positions_um,
	radius_um,
	lower_layer_indices=None,
	layer_weight_decay_lambda=1.0
):
	"""Create a CSR matrix mapping readout neurons to lower-layer neighbors.

	When ``lower_layer_indices`` is provided, each flattened neighbor has an
	associated source-layer index available for layer-dependent weighting.

	Weights are exponential in source-layer distance from the closest lower layer
	to readout (largest layer index):
		w ~ exp(-lambda * (max_layer - source_layer)).
	"""
	if radius_um <= 0:
		raise ValueError('radius_um must be positive.')

	lower_positions = np.asarray(lower_positions_um, dtype=np.float32)
	if lower_positions.ndim != 3 or lower_positions.shape[2] != 2:
		raise ValueError('lower_positions_um must have shape (n_layers, n_neurons, 2).')

	n_lower_layers = int(lower_positions.shape[0])
	n_lower_per_layer = int(lower_positions.shape[1])

	if lower_layer_indices is None:
		flat_lower_layer_ids = np.repeat(np.arange(n_lower_layers, dtype=np.int32), n_lower_per_layer)
	else:
		lower_layer_indices = np.asarray(lower_layer_indices, dtype=np.int32)
		if lower_layer_indices.shape[0] != n_lower_layers:
			raise ValueError(
				'lower_layer_indices length must match lower_positions_um first dimension.'
			)
		flat_lower_layer_ids = np.repeat(lower_layer_indices, n_lower_per_layer)

	max_layer_index = float(np.max(flat_lower_layer_ids))

	lower_flat = lower_positions.reshape(-1, 2)
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

		# Exponential decay by source-layer distance from readout-adjacent layer.
		neighbor_source_layers = flat_lower_layer_ids[in_radius].astype(np.float32)
		neighbor_layer_distance = max_layer_index - neighbor_source_layers
		neighbor_weights = np.exp(-float(layer_weight_decay_lambda) * neighbor_layer_distance)
		neighbor_weights_sum = float(np.sum(neighbor_weights))
		if neighbor_weights_sum <= 0.0:
			neighbor_weights = np.full(in_radius.size, 1.0 / float(in_radius.size), dtype=np.float32)
		else:
			neighbor_weights = (neighbor_weights / neighbor_weights_sum).astype(np.float32)

		rows.extend([readout_idx] * int(in_radius.size))
		cols.extend(in_radius.tolist())
		data.extend(neighbor_weights.tolist())

	n_readout = readout.shape[0]
	n_lower_total = lower_flat.shape[0]
	return sparse.csr_matrix((data, (rows, cols)), shape=(n_readout, n_lower_total), dtype=np.float32)


def compute_readout_tensor(voltage_lower_exc, neighborhood_csr):
	"""Compute mean neighborhood activation for each readout neuron at each time."""
	voltages = np.asarray(voltage_lower_exc, dtype=np.float32)
	n_time = voltages.shape[2]
	lower_flat_time = voltages.reshape(-1, n_time)
	readout_by_time = neighborhood_csr @ lower_flat_time
	return np.asarray(readout_by_time.T, dtype=np.float32)


def summarize_layer_contributions(neighborhood_csr, lower_layer_indices, n_lower_per_layer):
	"""Summarize how much neighborhood weight mass lands on each source layer."""
	if n_lower_per_layer <= 0:
		raise ValueError('n_lower_per_layer must be positive.')

	lower_layer_indices = np.asarray(lower_layer_indices, dtype=np.int32)
	if lower_layer_indices.ndim != 1:
		raise ValueError('lower_layer_indices must be a 1D array.')

	flat_layer_ids = np.repeat(lower_layer_indices, int(n_lower_per_layer))
	if flat_layer_ids.shape[0] != neighborhood_csr.shape[1]:
		raise ValueError(
			'Flattened layer IDs length does not match neighborhood matrix column count.'
		)

	weights_per_input = np.asarray(neighborhood_csr.sum(axis=0)).ravel().astype(np.float64)
	total_mass = float(np.sum(weights_per_input))

	rows_with_neighbors = np.diff(neighborhood_csr.indptr) > 0
	row_count = int(neighborhood_csr.shape[0])
	n_rows_with_neighbors = int(np.sum(rows_with_neighbors))

	layer_totals = {}
	for layer_id in np.unique(flat_layer_ids):
		mask = flat_layer_ids == layer_id
		mass = float(np.sum(weights_per_input[mask]))
		fraction = (mass / total_mass) if total_mass > 0 else np.nan
		layer_totals[int(layer_id)] = {
			'mass': mass,
			'fraction': fraction,
		}

	return {
		'row_count': row_count,
		'rows_with_neighbors': n_rows_with_neighbors,
		'total_mass': total_mass,
		'layer_totals': layer_totals,
	}


def generate_readout_tensor_from_file(
	radius_um,
	layer_weight_decay_lambda=1.0,
	input_npz='output/internal/lower_layer_voltage_raw.npz',
	output_npz='output/internal/readout_avg_radius.npz',
	neighborhood_npz='output/internal/readout_neighborhood_radius.npz',
):
	"""Load raw data, build neighborhood matrix, and write readout tensor to disk."""
	bundle = load_raw_voltage_bundle(input_npz)
	neighborhood = build_radius_neighborhood_matrix(
		bundle['lower_positions_um'],
		bundle['readout_positions_um'],
		radius_um=radius_um,
		lower_layer_indices=bundle['lower_layer_indices'],
		layer_weight_decay_lambda=layer_weight_decay_lambda,
	)
	readout_tensor = compute_readout_tensor(bundle['voltage_lower_exc'], neighborhood)

	Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
	sparse.save_npz(neighborhood_npz, neighborhood)
	np.savez_compressed(
		output_npz,
		readout_avg=readout_tensor,
		time_ms=bundle['time_ms'],
		radius_um=np.float32(radius_um),
		layer_weight_decay_lambda=np.float32(layer_weight_decay_lambda),
	)
	return bundle, readout_tensor, neighborhood


def assign_nearest_centroid_ids(positions_um, centroids):
	"""Assign each neuron to the closest centroid in Euclidean distance."""
	centroids_um = np.array([[float(cx / um), float(cy / um)] for cx, cy in centroids], dtype=np.float32)
	diffs = positions_um[:, None, :] - centroids_um[None, :, :]
	dist2 = np.sum(diffs * diffs, axis=2)
	return np.argmin(dist2, axis=1).astype(np.int32)


def plot_proxy_column_raster(
	result,
	proxy_column_ids,
	readout_positions_um=None,
	readout_tensor=None,
	time_ms=None,
	output_path='output/public/neuropil_readout_raster.png',
):
	"""Save a spike raster with an optional lower-layer activation heatmap."""
	import matplotlib.pyplot as plt
	from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

	proxy_column_ids = np.asarray(proxy_column_ids, dtype=np.int32)
	n_exc = proxy_column_ids.size
	positions_um = None if readout_positions_um is None else np.asarray(readout_positions_um, dtype=np.float32)
	exc_sort_order = np.argsort(proxy_column_ids, kind='stable')
	exc_row_map = np.empty(n_exc, dtype=np.int32)
	exc_row_map[exc_sort_order] = np.arange(n_exc, dtype=np.int32)
	sorted_ids = proxy_column_ids[exc_sort_order]
	unique_ids, counts = np.unique(sorted_ids, return_counts=True)
	cmap = plt.get_cmap('tab10')

	exc_spike_i = np.asarray(result['spike_mon_exc'].i[:], dtype=np.int32)
	exc_spike_t_ms = np.asarray(result['spike_mon_exc'].t[:] / ms, dtype=np.float32)
	inh_spike_i = np.asarray(result['spike_mon_inh'].i[:], dtype=np.int32)
	inh_spike_t_ms = np.asarray(result['spike_mon_inh'].t[:] / ms, dtype=np.float32)
	exc_spike_rows = exc_row_map[exc_spike_i]
	exc_spike_cluster_ids = proxy_column_ids[exc_spike_i]
	exc_spike_colors = cmap(exc_spike_cluster_ids % 10)

	if readout_tensor is not None:
		if time_ms is None:
			raise ValueError('time_ms is required when readout_tensor is provided.')
		time_ms = np.asarray(time_ms, dtype=np.float32)
		if time_ms.shape[0] != np.asarray(readout_tensor).shape[0]:
			raise ValueError('time_ms and readout_tensor must have the same number of time samples.')
		if positions_um is not None:
			fig = plt.figure(figsize=(15.5, 11.0))
			gs = fig.add_gridspec(
				2,
				2,
				width_ratios=[3.3, 1.6],
				height_ratios=[1, 2],
				wspace=0.28,
				hspace=0.12,
			)
			ax_heat = fig.add_subplot(gs[0, 0])
			ax = fig.add_subplot(gs[1, 0], sharex=ax_heat)
			ax_spatial = fig.add_subplot(gs[:, 1])
		else:
			fig, (ax_heat, ax) = plt.subplots(
				2,
				1,
				figsize=(12, 11),
				sharex=True,
				gridspec_kw={'height_ratios': [1, 2]},
			)
			ax_spatial = None
		activation = np.asarray(readout_tensor, dtype=np.float32)[:, exc_sort_order].T
		act_min = float(np.min(activation))
		act_max = float(np.max(activation))

		if act_min < 0.0:
			abs_lim = float(max(abs(act_min), abs(act_max)))
			if abs_lim <= 0.0:
				abs_lim = 1.0
			activation_cmap = LinearSegmentedColormap.from_list(
				'blue_white_red',
				['#0b4cc2', '#ffffff', '#b40426'],
			)
			norm = TwoSlopeNorm(vmin=-abs_lim, vcenter=0.0, vmax=abs_lim)
			cbar_label = 'Activation (signed, symmetric scale)'
		else:
			activation_cmap = LinearSegmentedColormap.from_list('white_dark_red', ['#ffffff', '#8b0000'])
			max_lim = act_max if act_max > 0.0 else 1.0
			norm = Normalize(vmin=0.0, vmax=max_lim)
			cbar_label = 'Activation'

		im = ax_heat.imshow(
			activation,
			aspect='auto',
			origin='lower',
			interpolation='nearest',
			extent=[float(time_ms[0]), float(time_ms[-1]), 0.0, float(n_exc)],
			cmap=activation_cmap,
			norm=norm,
		)
		ax_heat.set_ylabel('Neuron index')
		ax_heat.set_title('Lower-layer activation received by readout neurons from avg. field')
		ax_heat.set_ylim(0.0, float(n_exc))
		ax_heat.set_yticks([])

		y_cursor = 0
		for cluster_idx, count in zip(unique_ids, counts):
			y_start = y_cursor
			y_end = y_cursor + int(count)
			band_color = cmap(int(cluster_idx) % 10)
			ax_heat.axhline(y_start, color=band_color, linewidth=0.6, alpha=0.75)
			ax_heat.text(
				1.01,
				0.5 * (y_start + y_end),
				f'C{int(cluster_idx)}',
				transform=ax_heat.get_yaxis_transform(),
				fontsize=8,
				color=band_color,
				ha='left',
				va='center',
				clip_on=False,
			)
			y_cursor = y_end
		fig.colorbar(im, ax=ax_heat, pad=0.01, fraction=0.04, label=cbar_label)
	else:
		if positions_um is not None:
			fig, (ax, ax_spatial) = plt.subplots(
				1,
				2,
				figsize=(15.5, 8.0),
				gridspec_kw={'width_ratios': [3.1, 1.7]},
			)
		else:
			fig, ax = plt.subplots(figsize=(12, 8))
			ax_spatial = None

	ax.scatter(exc_spike_t_ms, exc_spike_rows, s=2, c=exc_spike_colors, alpha=0.7, label='Excitatory')
	ax.scatter(inh_spike_t_ms, n_exc + inh_spike_i, s=2, c='tab:blue', alpha=0.7, label='Inhibitory')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Neuron index')
	ax.set_title('Neuropil readout layer: spike raster by proxy column')
	ax.legend(loc='upper right', markerscale=3)

	y_cursor = 0
	for block_idx, (cluster_idx, count) in enumerate(zip(unique_ids, counts)):
		y_start = y_cursor
		y_end = y_cursor + int(count)
		y_center = 0.5 * (y_start + y_end - 1)
		band_color = cmap(int(cluster_idx) % 10)
		ax.text(
			1.02,
			y_center,
			f'C{int(cluster_idx)}',
			transform=ax.get_yaxis_transform(),
			fontsize=8,
			color=band_color,
			ha='left',
			va='center',
			clip_on=False,
		)
		y_cursor = y_end

	ax.axhline(n_exc, color='gray', linewidth=0.6, alpha=0.8)
	ax.text(1.02, 0.5 * n_exc, 'readout', transform=ax.get_yaxis_transform(), fontsize=8, color='tab:red', ha='left', va='center', clip_on=False)
	ax.text(1.02, n_exc + 0.5 * N_inh, 'inh', transform=ax.get_yaxis_transform(), fontsize=8, color='tab:blue', ha='left', va='center', clip_on=False)
	ax.set_xlim(0.0, float(duration / ms))
	ax.set_ylim(-1, n_exc + N_inh + 1)

	if ax_spatial is not None:
		for cluster_idx in np.unique(proxy_column_ids):
			mask = proxy_column_ids == cluster_idx
			color = cmap(int(cluster_idx) % 10)
			ax_spatial.scatter(
				positions_um[mask, 0],
				positions_um[mask, 1],
				s=12,
				alpha=0.75,
				color=color,
				edgecolors='none',
				label=f'C{int(cluster_idx)}',
			)
			if np.any(mask):
				mx = float(np.mean(positions_um[mask, 0]))
				my = float(np.mean(positions_um[mask, 1]))
				ax_spatial.text(mx, my, f'C{int(cluster_idx)}', fontsize=9, color='black', ha='center', va='center')

		ax_spatial.set_aspect('equal', adjustable='box')
		ax_spatial.set_xlabel('x (um)')
		ax_spatial.set_ylabel('y (um)')
		ax_spatial.set_title('Readout spatial layout (proxy-column colors)')
		ax_spatial.grid(alpha=0.2)
		ax_spatial.legend(loc='upper right', fontsize=8)

	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout(rect=(0.0, 0.0, 0.96, 1.0))
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def plot_proxy_column_spatial_band_mapping(
	proxy_column_ids,
	readout_positions_um,
	output_path='output/public/neuropil_proxy_column_mapping.png',
):
	"""Plot explicit spatial-to-raster-band mapping for the readout proxy columns."""
	proxy_column_ids = np.asarray(proxy_column_ids, dtype=np.int32)
	positions_um = np.asarray(readout_positions_um, dtype=np.float32)
	n_exc = int(proxy_column_ids.size)

	exc_sort_order = np.argsort(proxy_column_ids, kind='stable')
	sorted_ids = proxy_column_ids[exc_sort_order]
	unique_ids, counts = np.unique(sorted_ids, return_counts=True)
	cmap = plt.get_cmap('tab10')

	fig, (ax_spatial, ax_bands) = plt.subplots(
		1,
		2,
		figsize=(13, 5.5),
		gridspec_kw={'width_ratios': [1.2, 0.8]},
		constrained_layout=True,
	)

	for cluster_idx in np.unique(proxy_column_ids):
		mask = proxy_column_ids == cluster_idx
		color = cmap(int(cluster_idx) % 10)
		ax_spatial.scatter(
			positions_um[mask, 0],
			positions_um[mask, 1],
			s=12,
			alpha=0.72,
			color=color,
			edgecolors='none',
			label=f'C{int(cluster_idx)}',
		)
		if np.any(mask):
			mx = float(np.mean(positions_um[mask, 0]))
			my = float(np.mean(positions_um[mask, 1]))
			ax_spatial.text(mx, my, f'C{int(cluster_idx)}', fontsize=9, color='black', ha='center', va='center')

	ax_spatial.set_aspect('equal', adjustable='box')
	ax_spatial.set_xlabel('x (um)')
	ax_spatial.set_ylabel('y (um)')
	ax_spatial.set_title('Readout layer: proxy-column spatial regions')
	ax_spatial.grid(alpha=0.2)
	ax_spatial.legend(loc='upper right', fontsize=8)

	y_cursor = 0
	for cluster_idx, count in zip(unique_ids, counts):
		y_start = y_cursor
		y_end = y_cursor + int(count)
		color = cmap(int(cluster_idx) % 10)
		ax_bands.axhspan(y_start, y_end, color=color, alpha=0.25)
		ax_bands.text(0.5, 0.5 * (y_start + y_end), f'C{int(cluster_idx)}', ha='center', va='center', fontsize=10)
		ax_bands.axhline(y_start, color='black', linewidth=0.4, alpha=0.35)
		y_cursor = y_end
	ax_bands.axhline(n_exc, color='black', linewidth=0.6, alpha=0.6)
	ax_bands.set_xlim(0.0, 1.0)
	ax_bands.set_xticks([])
	ax_bands.set_ylim(0.0, float(n_exc))
	ax_bands.set_ylabel('Sorted excitatory row index')
	ax_bands.set_title('Raster band order (sorted by proxy column)')

	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def build_uniform_readout_layer(bundle, readout_tensor):
	"""Create a uniform readout layer driven by the precomputed readout tensor."""
	start_scope()
	seed(RANDOM_SEED)

	time_ms = np.asarray(bundle['time_ms'], dtype=np.float32)
	if readout_tensor.shape[0] != time_ms.shape[0]:
		raise ValueError(
			f"Readout tensor has {readout_tensor.shape[0]} time points but bundle has {time_ms.shape[0]}."
		)
	if readout_tensor.shape[1] != bundle['readout_positions_um'].shape[0]:
		raise ValueError(
			'Number of readout traces must match number of readout-layer excitatory neurons.'
		)

	if time_ms.shape[0] < 2:
		raise ValueError('Bundle must contain at least two time samples.')
	input_dt = float(time_ms[1] - time_ms[0]) * ms
	readout_drive = TimedArray(readout_tensor, dt=input_dt)

	readout_positions_um = np.asarray(bundle['readout_positions_um'], dtype=np.float32)
	n_readout = int(readout_positions_um.shape[0])
	uniform_radius = (R + 2 * sigma_c) / um
	proxy_centroids = pentacle_points(radius=R)
	proxy_column_ids = assign_nearest_centroid_ids(readout_positions_um, proxy_centroids)

	# Keep readout neuron indices contiguous by proxy column.
	sort_order = np.argsort(proxy_column_ids, kind='stable')
	readout_positions_um = readout_positions_um[sort_order]
	proxy_column_ids = proxy_column_ids[sort_order]
	readout_tensor = np.asarray(readout_tensor, dtype=np.float32)[:, sort_order]

	_, inh_positions_um, inh_cluster_ids = generate_uniform_layout(
		radius=uniform_radius,
		n_neurons=N_inh,
	)

	readout_exc_neurons = NeuronGroup(
		n_readout,
		eqs_exc_readout,
		threshold='v > v_th',
		reset='v = v_reset',
		refractory=refractory,
		method='euler',
		namespace={'readout_drive': readout_drive},
	)
	readout_inh_neurons = NeuronGroup(
		N_inh,
		eqs_inh_readout,
		threshold='v > v_th',
		reset='v = v_reset',
		refractory=refractory,
		method='euler',
	)

	readout_exc_neurons.tau_m = tau_m_e
	readout_exc_neurons.mu = '1.1 + 0.1*rand()'
	readout_exc_neurons.v = 'rand()'
	readout_exc_neurons.g_e = 0 * Hz
	readout_exc_neurons.g_i = 0 * Hz
	readout_exc_neurons.column_id = proxy_column_ids
	readout_exc_neurons.x = readout_positions_um[:, 0] * um
	readout_exc_neurons.y = readout_positions_um[:, 1] * um

	readout_inh_neurons.tau_m = tau_m_i
	readout_inh_neurons.mu = '1 + 0.05*rand()'
	readout_inh_neurons.v = 'rand()'
	readout_inh_neurons.g_e = 0 * Hz
	readout_inh_neurons.g_i = 0 * Hz
	readout_inh_neurons.column_id = inh_cluster_ids
	readout_inh_neurons.x = inh_positions_um[:, 0] * um
	readout_inh_neurons.y = inh_positions_um[:, 1] * um

	syn_ee = Synapses(readout_exc_neurons, readout_exc_neurons, model='w_syn : Hz', on_pre='g_e_post += w_syn')
	syn_ee.connect(condition='i != j', p=p_avg)
	syn_ee.w_syn = 'w_ee_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / weight_decay_l)'

	syn_ii = Synapses(readout_inh_neurons, readout_inh_neurons, model='w_syn : Hz', on_pre='g_i_post += w_syn')
	syn_ii.connect(condition='i != j', p=0.5)
	syn_ii.w_syn = 'w_ii_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / inh_weight_decay_l)'

	syn_ei = Synapses(readout_inh_neurons, readout_exc_neurons, model='w_syn : Hz', on_pre='g_i_post += w_syn')
	syn_ei.connect(p=0.5)
	syn_ei.w_syn = 'w_ei_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / inh_weight_decay_l)'

	syn_ie = Synapses(readout_exc_neurons, readout_inh_neurons, model='w_syn : Hz', on_pre='g_e_post += w_syn')
	syn_ie.connect(p=0.5)
	syn_ie.w_syn = 'w_ie_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / inh_weight_decay_l)'

	spike_mon_exc = SpikeMonitor(readout_exc_neurons)
	spike_mon_inh = SpikeMonitor(readout_inh_neurons)
	state_mon_exc = StateMonitor(readout_exc_neurons, 'v', record=True)

	net = Network([
		readout_exc_neurons,
		readout_inh_neurons,
		syn_ee,
		syn_ii,
		syn_ei,
		syn_ie,
		spike_mon_exc,
		spike_mon_inh,
		state_mon_exc,
	])
	net.run(duration)

	return {
		'exc_neurons': readout_exc_neurons,
		'inh_neurons': readout_inh_neurons,
		'syn_ee': syn_ee,
		'syn_ii': syn_ii,
		'syn_ei': syn_ei,
		'syn_ie': syn_ie,
		'spike_mon_exc': spike_mon_exc,
		'spike_mon_inh': spike_mon_inh,
		'state_mon_exc': state_mon_exc,
		'proxy_column_ids': proxy_column_ids,
	}


def save_readout_simulation_outputs(
	result,
	readout_tensor,
	bundle,
	radius_um,
	output_npz='output/internal/readout_layer_simulation.npz',
):
	"""Persist the computed readout drive and readout-layer response."""
	time_ms = np.asarray(bundle['time_ms'], dtype=np.float32)
	readout_v = np.asarray(result['state_mon_exc'].v[:], dtype=np.float32)
	exc_spike_i = np.asarray(result['spike_mon_exc'].i[:], dtype=np.int32)
	exc_spike_t_ms = np.asarray(result['spike_mon_exc'].t[:] / ms, dtype=np.float32)
	inh_spike_i = np.asarray(result['spike_mon_inh'].i[:], dtype=np.int32)
	inh_spike_t_ms = np.asarray(result['spike_mon_inh'].t[:] / ms, dtype=np.float32)

	Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		output_npz,
		readout_avg=readout_tensor,
		readout_exc_voltage=readout_v,
		time_ms=time_ms,
		radius_um=np.float32(radius_um),
		n_readout_exc=np.int32(readout_v.shape[0]),
		duration_ms=np.float32(float(duration / ms)),
		readout_positions_um=np.asarray(bundle['readout_positions_um'], dtype=np.float32),
		exc_spike_i=exc_spike_i,
		exc_spike_t_ms=exc_spike_t_ms,
		inh_spike_i=inh_spike_i,
		inh_spike_t_ms=inh_spike_t_ms,
		proxy_column_ids=np.asarray(result['proxy_column_ids'], dtype=np.int32),
	)


def main(
	radius_um=10.0,
	layer_weight_decay_lambda=1.0,
	mi_bin_width_ms=10.0,
	output_public_dir='output/public',
	output_internal_dir='output/internal',
):
	output_public_dir = Path(output_public_dir)
	output_internal_dir = Path(output_internal_dir)
	output_public_dir.mkdir(parents=True, exist_ok=True)
	output_internal_dir.mkdir(parents=True, exist_ok=True)

	bundle, readout_tensor, neighborhood = generate_readout_tensor_from_file(
		radius_um=radius_um,
		layer_weight_decay_lambda=layer_weight_decay_lambda,
		input_npz=output_internal_dir / 'lower_layer_voltage_raw.npz',
		output_npz=output_internal_dir / 'readout_avg_radius.npz',
		neighborhood_npz=output_internal_dir / 'readout_neighborhood_radius.npz',
	)
	result = build_uniform_readout_layer(bundle, readout_tensor)
	save_readout_simulation_outputs(
		result,
		readout_tensor,
		bundle,
		radius_um,
		output_npz=output_internal_dir / 'readout_layer_simulation.npz',
	)
	mi_matrix_neuropil = save_mi_outputs(
		spike_mon=result['spike_mon_exc'],
		bin_width_ms=mi_bin_width_ms,
		matrix_output_npz=str(output_internal_dir / 'mi_matrix_neuropil_readout_exc.npz'),
		heatmap_output_png=str(output_public_dir / 'mi_heatmap_neuropil_readout_exc.png'),
		title='Neuropil readout excitatory MI heatmap',
	)
	plot_proxy_column_raster(
		result,
		result['proxy_column_ids'],
		readout_positions_um=bundle['readout_positions_um'],
		readout_tensor=readout_tensor,
		time_ms=bundle['time_ms'],
		output_path=output_public_dir / 'neuropil_readout_raster.png',
	)
	plot_proxy_column_spatial_band_mapping(
		result['proxy_column_ids'],
		bundle['readout_positions_um'],
		output_path=output_public_dir / 'neuropil_proxy_column_mapping.png',
	)

	neighbor_counts = np.diff(neighborhood.indptr)
	lower_layer_indices = np.asarray(bundle['lower_layer_indices'], dtype=np.int32)
	n_lower_per_layer = int(np.asarray(bundle['lower_positions_um']).shape[1])
	layer_summary = summarize_layer_contributions(
		neighborhood,
		lower_layer_indices=lower_layer_indices,
		n_lower_per_layer=n_lower_per_layer,
	)

	print(f'Loaded raw bundle with lower voltages shape {bundle["voltage_lower_exc"].shape}.')
	print(f'Computed readout tensor with shape {readout_tensor.shape}.')
	print(f'Layer-weight decay lambda: {layer_weight_decay_lambda:.4f}')
	print(
		'Readout neighborhood counts: '
		f'min={neighbor_counts.min()}, mean={neighbor_counts.mean():.1f}, max={neighbor_counts.max()}'
	)
	print(
		'Neighborhood coverage: '
		f"rows_with_neighbors={layer_summary['rows_with_neighbors']}/{layer_summary['row_count']}, "
		f"total_weight_mass={layer_summary['total_mass']:.3f}"
	)
	for layer_id in sorted(layer_summary['layer_totals']):
		layer_mass = layer_summary['layer_totals'][layer_id]['mass']
		layer_fraction = layer_summary['layer_totals'][layer_id]['fraction']
		print(
			f"  Source layer {layer_id}: mass={layer_mass:.6f}, "
			f"fraction={layer_fraction:.6f}"
		)
	print(
		'Readout-layer spikes: '
		f"E={result['spike_mon_exc'].num_spikes}, I={result['spike_mon_inh'].num_spikes}"
	)
	print(
		'Saved neuropil MI matrix '
		f"{mi_matrix_neuropil.shape} -> {output_internal_dir / 'mi_matrix_neuropil_readout_exc.npz'}"
	)
	print(f"Saved: {output_public_dir / 'neuropil_readout_raster.png'}")
	print(f"Saved: {output_public_dir / 'neuropil_proxy_column_mapping.png'}")
	print(f"Saved: {output_public_dir / 'mi_heatmap_neuropil_readout_exc.png'}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate neuropil readout drive and simulate a readout layer.')
	parser.add_argument('--radius-um', type=float, default=25.0, help='Neighborhood radius in micrometers.')
	parser.add_argument(
		'--layer-weight-decay-lambda',
		type=float,
		default=1.0,
		help='Exponential decay rate by source-layer distance (larger => stronger preference for readout-adjacent lower layers).',
	)
	parser.add_argument(
		'--mi-bin-width-ms',
		type=float,
		default=10.0,
		help='Bin width (ms) for mutual information matrix construction.',
	)
	parser.add_argument(
		'--duration-ms',
		type=float,
		default=2000.0,
		help='Simulation duration in milliseconds.',
	)
	parser.add_argument(
		'--output-public-dir',
		default='output/public',
		help='Directory for public figure/text outputs.',
	)
	parser.add_argument(
		'--output-internal-dir',
		default='output/internal',
		help='Directory for internal intermediate outputs.',
	)
	args = parser.parse_args()
	if args.mi_bin_width_ms <= 0.0:
		raise ValueError('--mi-bin-width-ms must be positive.')
	if args.duration_ms <= 0.0:
		raise ValueError('--duration-ms must be positive.')
	set_simulation_duration_ms(args.duration_ms)
	main(
		radius_um=args.radius_um,
		layer_weight_decay_lambda=args.layer_weight_decay_lambda,
		mi_bin_width_ms=args.mi_bin_width_ms,
		output_public_dir=args.output_public_dir,
		output_internal_dir=args.output_internal_dir,
	)
