from brian2 import *

RANDOM_SEED = 2
# Simulation settings
duration = 5 * second
defaultclock.dt = 0.1 * ms

N_exc_c = 320
N_inh = 400 
R = 40 * um  # Radius of the pentacle
sigma_c = 20 * um  # radius for Gaussian dist of assembly neurons
sigma_connection = 10 * um

# Parameters copied from SSA paper https://arxiv.org/abs/1502.05656
tau_m_e = 15 * ms
tau_m_i = 10 * ms
tau_e = 3 * ms
tau_i = 2 * ms
v_reset = 0.0
v_th = 1.0
refractory = 5 * ms

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

def get_spatial_assembly_layout_target_p_avg(
	assembly_radius,
	pentacle_radius,
	sigma_connection,
	target_overall_avg=0.05,
	n_clusters=5,
	neurons_per_cluster=N_exc_c,
	tol=1e-6,
	max_iter=80,
):
	if target_overall_avg <= 0.0 or target_overall_avg >= 1.0:
		raise ValueError("target_overall_avg must be in (0, 1).")
	if float(sigma_connection / um) <= 0.0:
		raise ValueError("sigma_connection must be positive.")

	centroids, neuron_locations, positions_um, cluster_ids = generate_pentacle_layout(
		assembly_radius=assembly_radius,
		pentacle_radius=pentacle_radius,
		n_clusters=n_clusters,
		neurons_per_cluster=neurons_per_cluster,
	)

	dx = positions_um[:, 0][:, None] - positions_um[:, 0][None, :]
	dy = positions_um[:, 1][:, None] - positions_um[:, 1][None, :]
	dist_um = np.sqrt(dx * dx + dy * dy)
	base_kernel = np.exp(-dist_um / float(sigma_connection / um))
	mask_offdiag = ~np.eye(base_kernel.shape[0], dtype=bool)

	def avg_prob_for_pmax(p_max):
		return float(np.minimum(1.0, p_max * base_kernel)[mask_offdiag].mean())

	low = 0.0
	high = 1.0
	avg_high = avg_prob_for_pmax(high)
	while avg_high < target_overall_avg and high < 1e6:
		high *= 2.0
		avg_high = avg_prob_for_pmax(high)

	if avg_high < target_overall_avg:
		raise ValueError("Could not find p_max_exc that achieves target_overall_avg.")

	for _ in range(max_iter):
		mid = 0.5 * (low + high)
		avg_mid = avg_prob_for_pmax(mid)
		if abs(avg_mid - target_overall_avg) <= tol:
			p_max_exc = mid
			break
		if avg_mid < target_overall_avg:
			low = mid
		else:
			high = mid
	else:
		p_max_exc = 0.5 * (low + high)

	return p_max_exc, centroids, neuron_locations, positions_um, cluster_ids

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