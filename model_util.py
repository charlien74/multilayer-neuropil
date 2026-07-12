from brian2 import *

RANDOM_SEED = 2
# Simulation settings
duration = 2 * second
defaultclock.dt = 0.1 * ms

N_exc_c = 320
N_inh = 700 
R = 60 * um  # Radius of the pentacle
sigma_c = 20 * um  # radius for Gaussian dist of assembly neurons
sigma_connection = 10 * um
weight_decay_l = 500 * um

# Parameters copied from SSA paper https://arxiv.org/abs/1502.05656
tau_m_e = 15 * ms
tau_m_i = 10 * ms
tau_e = 25 * ms
tau_i = 15 * ms
v_reset = 0.0
v_th = 1.0
refractory = 8 * ms

eqs_exc = """
dv/dt = (mu - v) / tau_m + w_ee * g_e + w_ei * g_i : 1 (unless refractory)
dg_e/dt = -g_e / tau_e : 1
dg_i/dt = -g_i / tau_i : 1
mu : 1
tau_m : second (constant)
w_ee : Hz
w_ei : Hz
column_id : integer (constant)
x: meter
y: meter
"""

eqs_inh = """
dv/dt = (mu - v) / tau_m + w_ie * g_e + w_ii * g_i : 1 (unless refractory)
dg_e/dt = -g_e / tau_e : 1
dg_i/dt = -g_i / tau_i : 1
mu : 1
tau_m : second (constant)
w_ie : Hz
w_ii : Hz
column_id : integer (constant)
x: meter
y: meter
"""

def pentacle_points(radius, center=(0.0 * um, 0.0 * um), start_angle_deg=-90.0):
	"""Return 5 (x, y) points ordered to trace a pentacle (5-point star).

	The points lie on a circle of the given radius around ``center``.
	"""
	cx, cy = center
	base = np.deg2rad(start_angle_deg)

	# 5 equally spaced vertices on the circle.
	vertices = []
	for i in range(5):
		a = base + i * 2 * np.pi / 5
		x = cx + radius * np.cos(a)
		y = cy + radius * np.sin(a)
		vertices.append((x, y))

	return vertices 

def compute_connection_probability_averages(positions_um, cluster_ids, p_max, sigma_um):
	"""Compute mean p(d) for in-cluster, out-cluster, and all non-self pairs."""
	n = positions_um.shape[0]
	if n < 2:
		return np.nan, np.nan, np.nan

	dx = positions_um[:, 0][:, None] - positions_um[:, 0][None, :]
	dy = positions_um[:, 1][:, None] - positions_um[:, 1][None, :]
	dist_um = np.sqrt(dx * dx + dy * dy)
	prob = np.minimum(1.0, p_max * np.exp(-dist_um / sigma_um))

	mask_offdiag = ~np.eye(n, dtype=bool)
	same_cluster = cluster_ids[:, None] == cluster_ids[None, :]
	in_cluster_mask = mask_offdiag & same_cluster
	out_cluster_mask = mask_offdiag & (~same_cluster)

	in_cluster_avg = prob[in_cluster_mask].mean() if np.any(in_cluster_mask) else np.nan
	out_cluster_avg = prob[out_cluster_mask].mean() if np.any(out_cluster_mask) else np.nan
	overall_avg = prob[mask_offdiag].mean() if np.any(mask_offdiag) else np.nan

	return in_cluster_avg, out_cluster_avg, overall_avg

def generate_pentacle_layout(assembly_radius, pentacle_radius, n_clusters=5, neurons_per_cluster=N_exc_c):
	"""Generate centroid locations and sampled neuron positions for a pentacle layout."""
	centroids = pentacle_points(radius=pentacle_radius)
	neuron_locations = []
	cluster_index = []
	assembly_radius_um = float(assembly_radius / um)

	for cluster_idx in range(n_clusters):
		for _ in range(neurons_per_cluster):
			x = centroids[cluster_idx][0] + np.random.normal(0, assembly_radius_um) * um
			y = centroids[cluster_idx][1] + np.random.normal(0, assembly_radius_um) * um
			neuron_locations.append((x, y))
			cluster_index.append(cluster_idx)

	positions_um = np.column_stack((
		np.array([x / um for x, _ in neuron_locations], dtype=float),
		np.array([y / um for _, y in neuron_locations], dtype=float),
	))
	cluster_ids = np.array(cluster_index, dtype=int)
	return centroids, neuron_locations, positions_um, cluster_ids

def generate_lattice_layout(assembly_radius, centroids_distance=None, clusters_shape=(5, 4), neurons_per_cluster=80):
	"""Generate centroid locations and sampled neuron positions for a lattice layout."""
	if centroids_distance is None:
		centroids_distance = 4 * assembly_radius
	centroids = []
	for i in range(clusters_shape[0]):
		for j in range(clusters_shape[1]):
			x = (i - (clusters_shape[0] - 1) / 2) * centroids_distance
			y = (j - (clusters_shape[1] - 1) / 2) * centroids_distance
			centroids.append((x, y))
	neuron_locations = []
	cluster_index = []
	assembly_radius_um = float(assembly_radius / um)

	n_clusters = clusters_shape[0] * clusters_shape[1]
	for cluster_idx in range(n_clusters):
		for _ in range(neurons_per_cluster):
			x = centroids[cluster_idx][0] + np.random.normal(0, assembly_radius_um) * um
			y = centroids[cluster_idx][1] + np.random.normal(0, assembly_radius_um) * um
			neuron_locations.append((x, y))
			cluster_index.append(cluster_idx)

	positions_um = np.column_stack((
		np.array([x / um for x, _ in neuron_locations], dtype=float),
		np.array([y / um for _, y in neuron_locations], dtype=float),
	))
	cluster_ids = np.array(cluster_index, dtype=int)
	return centroids, neuron_locations, positions_um, cluster_ids

def generate_uniform_layout(radius, n_neurons=1600, center=(0.0 * um, 0.0 * um)):
	
	neuron_locations = []
	cluster_index = []

	for _ in range(n_neurons):
		x = np.random.uniform(center[0] / um - radius, center[0] / um + radius) * um
		y = np.random.uniform(center[1] / um - radius, center[1] / um + radius) * um
		neuron_locations.append((x, y))
		cluster_index.append(0)  # All neurons belong to a single cluster

	positions_um = np.column_stack((
		np.array([x / um for x, _ in neuron_locations], dtype=float),
		np.array([y / um for _, y in neuron_locations], dtype=float),
	))
	cluster_ids = np.array(cluster_index, dtype=int)
	return neuron_locations, positions_um, cluster_ids

def generate_inhibitory_locations(assembly_radius, n_inhibitory):
	"""Generate single circular assembly for inhibitory neurons"""
	neuron_locations = []
	cluster_index = []
	assembly_radius_um = float(assembly_radius / um)

	for _ in range(n_inhibitory):
		x = np.random.normal(0, assembly_radius_um) * um
		y = np.random.normal(0, assembly_radius_um) * um
		neuron_locations.append((x, y))
		cluster_index.append(-1)  # Inhibitory neurons have a separate cluster ID

	positions_um = np.column_stack((
		np.array([x / um for x, _ in neuron_locations], dtype=float),
		np.array([y / um for _, y in neuron_locations], dtype=float),
	))
	cluster_ids = np.array(cluster_index, dtype=int)
	return neuron_locations, positions_um, cluster_ids

def compute_S_from_groups(spike_i, spike_t_ms, groups, bin_edges_ms, n_neurons):
	"""Average over bins of std(spike counts across groups)."""
	n_groups = len(groups)
	n_bins = len(bin_edges_ms) - 1
	if n_groups == 0 or n_bins <= 0:
		return np.nan

	neuron_to_group = np.full(n_neurons, -1, dtype=int)
	for group_idx, neurons in enumerate(groups):
		neuron_to_group[np.asarray(neurons, dtype=int)] = group_idx

	spike_group = neuron_to_group[spike_i]
	valid = spike_group >= 0
	if not np.any(valid):
		return 0.0

	spike_group = spike_group[valid]
	spike_t_valid = spike_t_ms[valid]
	bin_idx = np.searchsorted(bin_edges_ms, spike_t_valid, side='right') - 1
	bin_valid = (bin_idx >= 0) & (bin_idx < n_bins)

	counts = np.zeros((n_groups, n_bins), dtype=float)
	np.add.at(counts, (spike_group[bin_valid], bin_idx[bin_valid]), 1)

	std_per_bin = np.std(counts, axis=0)
	return float(np.mean(std_per_bin))


def compute_S_metrics(spike_i, spike_t_ms, cluster_ids, group_size, bin_size_ms=100.0, n_shuffles=10):
	"""Compute S, <S_shuff>, and S-<S_shuff> from excitatory spikes."""
	n_neurons = len(cluster_ids)
	if n_neurons % group_size != 0:
		raise ValueError("Number of neurons must be divisible by group_size.")

	t_max = duration / ms
	bin_edges_ms = np.arange(0.0, t_max + bin_size_ms + 1e-9, bin_size_ms)

	unique_clusters = np.unique(cluster_ids)
	true_groups = [np.flatnonzero(cluster_ids == c) for c in unique_clusters]
	S = compute_S_from_groups(spike_i, spike_t_ms, true_groups, bin_edges_ms, n_neurons)

	n_groups = n_neurons // group_size
	S_shuff_vals = []
	for _ in range(n_shuffles):
		perm = np.random.permutation(n_neurons)
		shuff_groups = [perm[g * group_size:(g + 1) * group_size] for g in range(n_groups)]
		S_shuff_vals.append(compute_S_from_groups(spike_i, spike_t_ms, shuff_groups, bin_edges_ms, n_neurons))

	S_shuff_mean = float(np.mean(S_shuff_vals)) if len(S_shuff_vals) > 0 else np.nan
	return S, S_shuff_mean, S - S_shuff_mean

def get_p_connection_in_out(R_ee, N_excitatory, cluster_size,p_ee_avg=0.2):
    numerator = p_ee_avg * (N_excitatory - 1)
    denominator = R_ee * (cluster_size - 1) + (N_excitatory - cluster_size)
    p_ee_out = numerator / denominator
    p_ee_in = R_ee * p_ee_out
    if p_ee_in > 1:
        print(f"R_ee={R_ee} too high (p_ee_in={p_ee_in:.4f} > 1); clamping.")
        p_ee_in = 1.0
        p_ee_out = p_ee_in / R_ee
    print(f"R_ee={R_ee}: p_ee_in={p_ee_in:.4f}, p_ee_out={p_ee_out:.4f}")
    return p_ee_in, p_ee_out
