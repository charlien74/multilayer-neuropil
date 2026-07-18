import argparse
from pathlib import Path

import numpy as np
from brian2 import *
from scipy import sparse

from model_util import *


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


def build_radius_neighborhood_matrix(lower_positions_um, readout_positions_um, radius_um):
	"""Create a CSR matrix mapping readout neurons to lower-layer neighbors."""
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
		weight = np.float32(1.0 / in_radius.size) * 100
		rows.extend([readout_idx] * int(in_radius.size))
		cols.extend(in_radius.tolist())
		data.extend([weight] * int(in_radius.size))

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


def generate_readout_tensor_from_file(
	radius_um,
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
	)
	readout_tensor = compute_readout_tensor(bundle['voltage_lower_exc'], neighborhood)

	Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
	sparse.save_npz(neighborhood_npz, neighborhood)
	np.savez_compressed(
		output_npz,
		readout_avg=readout_tensor,
		time_ms=bundle['time_ms'],
		radius_um=np.float32(radius_um),
	)
	return bundle, readout_tensor, neighborhood


def assign_nearest_centroid_ids(positions_um, centroids):
	"""Assign each neuron to the closest centroid in Euclidean distance."""
	centroids_um = np.array([[float(cx / um), float(cy / um)] for cx, cy in centroids], dtype=np.float32)
	diffs = positions_um[:, None, :] - centroids_um[None, :, :]
	dist2 = np.sum(diffs * diffs, axis=2)
	return np.argmin(dist2, axis=1).astype(np.int32)


def plot_proxy_column_raster(result, proxy_column_ids, output_path='output/public/neuropil_readout_raster.png'):
	"""Save a spike raster with excitatory rows grouped by proxy column."""
	import matplotlib.pyplot as plt

	proxy_column_ids = np.asarray(proxy_column_ids, dtype=np.int32)
	n_exc = proxy_column_ids.size
	exc_sort_order = np.argsort(proxy_column_ids, kind='stable')
	exc_row_map = np.empty(n_exc, dtype=np.int32)
	exc_row_map[exc_sort_order] = np.arange(n_exc, dtype=np.int32)

	exc_spike_i = np.asarray(result['spike_mon_exc'].i[:], dtype=np.int32)
	exc_spike_t_ms = np.asarray(result['spike_mon_exc'].t[:] / ms, dtype=np.float32)
	inh_spike_i = np.asarray(result['spike_mon_inh'].i[:], dtype=np.int32)
	inh_spike_t_ms = np.asarray(result['spike_mon_inh'].t[:] / ms, dtype=np.float32)
	exc_spike_rows = exc_row_map[exc_spike_i]

	fig, ax = plt.subplots(figsize=(12, 8))
	ax.scatter(exc_spike_t_ms, exc_spike_rows, s=2, c='tab:red', alpha=0.7, label='Excitatory')
	ax.scatter(inh_spike_t_ms, n_exc + inh_spike_i, s=2, c='tab:blue', alpha=0.7, label='Inhibitory')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Neuron index')
	ax.set_title('Neuropil readout layer: spike raster by proxy column')
	ax.legend(loc='upper right', markerscale=3)

	sorted_ids = proxy_column_ids[exc_sort_order]
	unique_ids, counts = np.unique(sorted_ids, return_counts=True)
	y_cursor = 0
	for block_idx, (cluster_idx, count) in enumerate(zip(unique_ids, counts)):
		y_start = y_cursor
		y_end = y_cursor + int(count)
		y_center = 0.5 * (y_start + y_end - 1)
		if block_idx % 2 == 0:
			ax.axhspan(y_start, y_end, color='gray', alpha=0.04)
		ax.text(
			1.02,
			y_center,
			f'C{int(cluster_idx)}',
			transform=ax.get_yaxis_transform(),
			fontsize=8,
			color='black',
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

	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout(rect=(0.0, 0.0, 0.96, 1.0))
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
		readout_positions_um=np.asarray(bundle['readout_positions_um'], dtype=np.float32),
		exc_spike_i=exc_spike_i,
		exc_spike_t_ms=exc_spike_t_ms,
		inh_spike_i=inh_spike_i,
		inh_spike_t_ms=inh_spike_t_ms,
		proxy_column_ids=np.asarray(result['proxy_column_ids'], dtype=np.int32),
	)


def main(radius_um=25.0):
	bundle, readout_tensor, neighborhood = generate_readout_tensor_from_file(radius_um=radius_um)
	result = build_uniform_readout_layer(bundle, readout_tensor)
	save_readout_simulation_outputs(result, readout_tensor, bundle, radius_um)
	plot_proxy_column_raster(result, result['proxy_column_ids'])

	neighbor_counts = np.diff(neighborhood.indptr)
	print(f'Loaded raw bundle with lower voltages shape {bundle["voltage_lower_exc"].shape}.')
	print(f'Computed readout tensor with shape {readout_tensor.shape}.')
	print(
		'Readout neighborhood counts: '
		f'min={neighbor_counts.min()}, mean={neighbor_counts.mean():.1f}, max={neighbor_counts.max()}'
	)
	print(
		'Readout-layer spikes: '
		f"E={result['spike_mon_exc'].num_spikes}, I={result['spike_mon_inh'].num_spikes}"
	)
	print('Saved: output/public/neuropil_readout_raster.png')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate neuropil readout drive and simulate a readout layer.')
	parser.add_argument('--radius-um', type=float, default=25.0, help='Neighborhood radius in micrometers.')
	args = parser.parse_args()
	main(radius_um=args.radius_um)
