from brian2 import SpikeMonitor, ms
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from util.spike_mi import *


def get_spike_monitor_positions_um(spike_mon: SpikeMonitor) -> np.ndarray:
    """Return neuron positions from a SpikeMonitor source as shape (N, 2) in um."""
    source = spike_mon.source
    if not hasattr(source, 'x') or not hasattr(source, 'y'):
        raise ValueError('SpikeMonitor source must expose x and y coordinates.')
    x_um = np.asarray(source.x[:] / source.x.unit, dtype=float)
    y_um = np.asarray(source.y[:] / source.y.unit, dtype=float)
    return np.column_stack((x_um, y_um))


def build_binned_spike_counts(spike_mon: SpikeMonitor,
                              dt_ms: float,
                              t_start_ms: float | None = None,
                              t_stop_ms: float | None = None) -> tuple[np.ndarray, float, float, float]:
    """Bin spikes into a dense occupancy matrix (N neurons x T bins)."""
    if dt_ms <= 0:
        raise ValueError('dt_ms must be > 0.')

    n_neurons = len(spike_mon.count)
    spike_t_ms = np.asarray(spike_mon.t / ms, dtype=float)

    if t_start_ms is None:
        t_start_ms = float(spike_t_ms.min()) if spike_t_ms.size else 0.0
    if t_stop_ms is None:
        t_stop_ms = float(spike_t_ms.max()) if spike_t_ms.size else float(t_start_ms + dt_ms)
    if t_stop_ms <= t_start_ms:
        t_stop_ms = t_start_ms + dt_ms

    spike_train_dict = spike_mon.spike_trains()
    spike_time_list = [
        np.asarray(spike_train_dict[i] / ms, dtype=np.float64)
        for i in range(n_neurons)
    ]
    counts = bin_population(
        spike_time_list,
        t_start=t_start_ms,
        t_stop=t_stop_ms,
        dt=dt_ms,
        clip=1,
    ).astype(np.float32)

    return counts, float(t_start_ms), float(t_stop_ms), float(dt_ms)


def split_count_matrix_into_windows(counts: np.ndarray,
                                    window_ms: float,
                                    dt_ms: float) -> np.ndarray:
    """Split count matrix (N x T) into windows of shape (W, N, Tw)."""
    if window_ms <= 0:
        raise ValueError('window_ms must be > 0.')
    if dt_ms <= 0:
        raise ValueError('dt_ms must be > 0.')

    bins_per_window = int(np.round(window_ms / dt_ms))
    if bins_per_window < 1:
        raise ValueError('window_ms must be at least one dt_ms bin.')

    n_neurons, n_bins = counts.shape
    n_windows = n_bins // bins_per_window
    if n_windows < 2:
        raise ValueError('Need at least two full windows to estimate distance-based MI.')

    trimmed = counts[:, :n_windows * bins_per_window]
    windows = trimmed.reshape(n_neurons, n_windows, bins_per_window).transpose(1, 0, 2)
    return np.asarray(windows, dtype=np.float32)


def build_spatial_kernel(positions_um: np.ndarray,
                         spatial_sigma_um: float) -> np.ndarray:
    """Build row-normalized squared-exponential spatial kernel matrix."""
    if spatial_sigma_um <= 0:
        raise ValueError('spatial_sigma_um must be > 0.')

    positions = np.asarray(positions_um, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError('positions_um must have shape (N, 2).')

    dx = positions[:, 0][:, None] - positions[:, 0][None, :]
    dy = positions[:, 1][:, None] - positions[:, 1][None, :]
    sqdist = dx * dx + dy * dy

    k = np.exp(-sqdist / float(spatial_sigma_um * spatial_sigma_um), dtype=np.float32)
    row_sums = np.sum(k, axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return (k / row_sums).astype(np.float32)


def apply_temporal_kernel(window_counts: np.ndarray,
                          dt_ms: float,
                          tau_ms: float) -> np.ndarray:
    """Apply causal exponential temporal smoothing to one window (N x T)."""
    if tau_ms <= 0:
        raise ValueError('tau_ms must be > 0.')

    counts = np.asarray(window_counts, dtype=np.float32)
    smoothed = np.zeros_like(counts)
    alpha = float(np.exp(-dt_ms / tau_ms))

    smoothed[:, 0] = counts[:, 0]
    for t in range(1, counts.shape[1]):
        smoothed[:, t] = alpha * smoothed[:, t - 1] + counts[:, t]
    return smoothed


def build_spatiotemporal_window_features(spike_mon: SpikeMonitor,
                                         window_ms: float,
                                         dt_ms: float,
                                         tau_ms: float,
                                         spatial_sigma_um: float,
                                         t_start_ms: float | None = None,
                                         t_stop_ms: float | None = None) -> dict:
    """Build flattened FA(x,t)-like features per window from a SpikeMonitor."""
    positions_um = get_spike_monitor_positions_um(spike_mon)
    counts, t0, t1, dt = build_binned_spike_counts(
        spike_mon,
        dt_ms=dt_ms,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
    )
    windows = split_count_matrix_into_windows(counts, window_ms=window_ms, dt_ms=dt)
    k_spatial = build_spatial_kernel(positions_um, spatial_sigma_um=spatial_sigma_um)

    feats = []
    for w in windows:
        temporal = apply_temporal_kernel(w, dt_ms=dt, tau_ms=tau_ms)
        spatial_temporal = k_spatial @ temporal
        feats.append(spatial_temporal.reshape(-1))

    features = np.asarray(feats, dtype=np.float32)
    return {
        'features': features,
        'positions_um': positions_um,
        'dt_ms': float(dt),
        'window_ms': float(window_ms),
        't_start_ms': float(t0),
        't_stop_ms': float(t1),
    }


def pairwise_l2_distance_matrix(features: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between feature vectors."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('features must be a 2D array (samples x dims).')
    gram = x @ x.T
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2, dtype=np.float64)


def k_nearest_neighbor_sets(distance_matrix: np.ndarray,
                            h: int) -> list[set[int]]:
    """Return k-nearest-neighbor index sets per sample (excluding self)."""
    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError('distance_matrix must be square.')
    if h <= 0:
        raise ValueError('h must be > 0.')

    n = d.shape[0]
    k = min(h, n - 1)
    neighbors = []
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        idx = np.argpartition(row, k)[:k]
        neighbors.append(set(int(v) for v in idx))
    return neighbors


def estimate_distance_based_mi_from_distance_matrices(distance_a: np.ndarray,
                                                      distance_b: np.ndarray,
                                                      h: int,
                                                      pseudo_count: int = 1) -> dict:
    """Estimate MI using nearest-neighbor set intersections (Houghton-style)."""
    da = np.asarray(distance_a, dtype=np.float64)
    db = np.asarray(distance_b, dtype=np.float64)
    if da.shape != db.shape:
        raise ValueError('distance_a and distance_b must have the same shape.')
    if da.ndim != 2 or da.shape[0] != da.shape[1]:
        raise ValueError('distance matrices must be square.')

    n = da.shape[0]
    if n < 2:
        raise ValueError('Need at least two samples/windows for MI estimation.')
    if pseudo_count < 0:
        raise ValueError('pseudo_count must be >= 0.')

    neigh_a = k_nearest_neighbor_sets(da, h=h)
    neigh_b = k_nearest_neighbor_sets(db, h=h)
    k_eff = min(h, n - 1)

    intersections = np.zeros(n, dtype=np.int32)
    for i in range(n):
        intersections[i] = len(neigh_a[i].intersection(neigh_b[i]))

    adjusted = intersections.astype(np.float64) + float(pseudo_count)
    terms = np.log2((n * adjusted) / float(k_eff * k_eff))
    mi_bits = float(np.mean(terms))

    return {
        'mi_bits': mi_bits,
        'terms_bits': terms,
        'intersection_counts': intersections,
        'k_effective': int(k_eff),
        'n_samples': int(n),
    }


def estimate_distance_based_mi_from_spike_monitors(spike_mon_a: SpikeMonitor,
                                                   spike_mon_b: SpikeMonitor,
                                                   window_ms: float,
                                                   dt_ms: float,
                                                   tau_ms: float,
                                                   spatial_sigma_um: float,
                                                   h: int,
                                                   pseudo_count: int = 1,
                                                   t_start_ms: float | None = None,
                                                   t_stop_ms: float | None = None) -> dict:
    """End-to-end distance-based MI estimate between two SpikeMonitor recordings."""
    if t_start_ms is None or t_stop_ms is None:
        ta = np.asarray(spike_mon_a.t / ms, dtype=float)
        tb = np.asarray(spike_mon_b.t / ms, dtype=float)
        if t_start_ms is None:
            starts = []
            if ta.size:
                starts.append(float(ta.min()))
            if tb.size:
                starts.append(float(tb.min()))
            t_start_ms = min(starts) if starts else 0.0
        if t_stop_ms is None:
            stops = []
            if ta.size:
                stops.append(float(ta.max()))
            if tb.size:
                stops.append(float(tb.max()))
            t_stop_ms = max(stops) if stops else float(t_start_ms + dt_ms)

    feats_a = build_spatiotemporal_window_features(
        spike_mon_a,
        window_ms=window_ms,
        dt_ms=dt_ms,
        tau_ms=tau_ms,
        spatial_sigma_um=spatial_sigma_um,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
    )
    feats_b = build_spatiotemporal_window_features(
        spike_mon_b,
        window_ms=window_ms,
        dt_ms=dt_ms,
        tau_ms=tau_ms,
        spatial_sigma_um=spatial_sigma_um,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
    )

    n_windows = min(feats_a['features'].shape[0], feats_b['features'].shape[0])
    if n_windows < 2:
        raise ValueError('Need at least two windows shared by both monitors.')

    fa = feats_a['features'][:n_windows]
    fb = feats_b['features'][:n_windows]

    d_a = pairwise_l2_distance_matrix(fa)
    d_b = pairwise_l2_distance_matrix(fb)
    est = estimate_distance_based_mi_from_distance_matrices(
        d_a,
        d_b,
        h=h,
        pseudo_count=pseudo_count,
    )

    return {
        'mi_bits': est['mi_bits'],
        'terms_bits': est['terms_bits'],
        'intersection_counts': est['intersection_counts'],
        'k_effective': est['k_effective'],
        'n_windows': int(n_windows),
        'distance_matrix_a': d_a,
        'distance_matrix_b': d_b,
        'features_a': fa,
        'features_b': fb,
        'params': {
            'window_ms': float(window_ms),
            'dt_ms': float(dt_ms),
            'tau_ms': float(tau_ms),
            'spatial_sigma_um': float(spatial_sigma_um),
            'h': int(h),
            'pseudo_count': int(pseudo_count),
            't_start_ms': float(t_start_ms),
            't_stop_ms': float(t_stop_ms),
        },
    }

def bin_spikes(spike_mon: SpikeMonitor,
               bin_width_ms: float) -> np.ndarray:
    if bin_width_ms <= 0:
        raise ValueError("bin_width_ms must be > 0.")

    n_neurons = len(spike_mon.count)
    spike_neuron_ids = np.asarray(spike_mon.i, dtype=np.int64)
    spike_times_ms = np.asarray(spike_mon.t / ms, dtype=float)

    if spike_times_ms.size == 0:
        return np.zeros((n_neurons, 1), dtype=np.int32)

    n_bins = int(np.floor(spike_times_ms.max() / bin_width_ms)) + 1
    binned_spikes = np.zeros((n_neurons, n_bins), dtype=np.int32)

    bin_ids = np.floor(spike_times_ms / bin_width_ms).astype(np.int64)
    valid = (
        (spike_neuron_ids >= 0)
        & (spike_neuron_ids < n_neurons)
        & (bin_ids >= 0)
        & (bin_ids < n_bins)
    )
    np.add.at(binned_spikes, (spike_neuron_ids[valid], bin_ids[valid]), 1)

    return binned_spikes

def compute_mi_binary(x: np.ndarray, y: np.ndarray,
                      entropy_normalize: bool = False) -> float:
    """
    Compute mutual information between two spike trains
    """
    if x.shape != y.shape:
        raise ValueError("Spike trains must have the same length.")
    if not np.isin(x, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Inputs must contain only 0 and 1.")
    
    joint = np.zeros((2, 2), dtype=float)

    for a, b in zip(x, y):
        joint[a, b] += 1

    joint /= len(x)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for a in range(2):
        for b in range(2):
            if joint[a, b] > 0:
                mi += joint[a, b] * np.log2(
                    joint[a, b] / (px[a] * py[b])
                )

    if entropy_normalize:
        h_x = -np.sum(px * np.log2(px + 1e-12))
        h_y = -np.sum(py * np.log2(py + 1e-12))
        mi /= max(h_x, h_y)
    return float(mi)

def compute_mi_matrix(spike_trains: np.ndarray, entropy_normalize: bool = False) -> np.ndarray:
    """
    Compute mutual information matrix for a set of spike trains
    """
    n_neurons = spike_trains.shape[0]
    mi_matrix = np.zeros((n_neurons, n_neurons), dtype=float)

    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            mi = compute_mi_binary(spike_trains[i], 
                                   spike_trains[j], 
                                   entropy_normalize=entropy_normalize)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def bin_spikes_and_compute_mi_matrix(spike_mon: SpikeMonitor,
                                     bin_width_ms: float,
                                     entropy_normalize: bool = False) -> np.ndarray:
    """Bin spikes, binarize occupancy, then compute pairwise MI matrix."""
    spike_counts = bin_spikes(spike_mon=spike_mon, bin_width_ms=bin_width_ms)
    spike_binary = (spike_counts > 0).astype(np.int8)
    return compute_mi_matrix(spike_binary, entropy_normalize=entropy_normalize)


def bin_spikes_and_compute_mi_matrix_corrected(spike_mon: SpikeMonitor,
                                               bin_width_ms: float,
                                               n_null: int = 100,
                                               lag: int = 10,
                                               method: str = 'plugin'):
    spike_times_ms = np.asarray(spike_mon.t / ms, dtype=float)
    t_start = spike_times_ms.min() if spike_times_ms.size > 0 else 0.0
    t_stop = spike_times_ms.max() if spike_times_ms.size > 0 else 0.0
    spike_train_dict = spike_mon.spike_trains()
    n_excitatory = len(spike_train_dict.keys())

    spike_time_list = [
        np.asarray(spike_train_dict[i] / ms)
        for i in range(n_excitatory)
    ]

    binned_spikes = bin_population(spike_time_list,
                                   t_start=t_start,
                                   t_stop=t_stop,
                                   dt=bin_width_ms)

    res = mi_pairs(
        binned_spikes,
        n_null=n_null,
        lag=lag,
    )

    mi_matrix = np.zeros((n_excitatory, n_excitatory), dtype=float)
    pairs = np.asarray(res['pairs'], dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError('res[\'pairs\'] must have shape (P, 2).')

    mi_corrected = np.asarray(res['mi_corrected'], dtype=float)
    methods = tuple(res.get('methods', ()))

    if mi_corrected.ndim == 2:
        if method in methods:
            method_idx = methods.index(method)
        else:
            method_idx = 0
        mi_values = mi_corrected[:, method_idx]
    elif mi_corrected.ndim == 1:
        mi_values = mi_corrected
    else:
        raise ValueError('res[\'mi_corrected\'] must be 1D or 2D.')

    if mi_values.shape[0] != pairs.shape[0]:
        raise ValueError('Pair count and MI value count do not match.')

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    mi_matrix[i_idx, j_idx] = mi_values
    mi_matrix[j_idx, i_idx] = mi_values
    return mi_matrix


def save_mi_heatmap(mi_matrix: np.ndarray,
                    output_path: str,
                    lognorm: bool = False,
                    title: str = 'Mutual Information Heatmap') -> None:
    """Save a red-scale MI heatmap where darker red means larger MI."""
    matrix = np.asarray(mi_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('mi_matrix must be a square 2D array.')

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    if lognorm:
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        im = ax.imshow(
            matrix,
            cmap="Reds",
            norm=LogNorm(
                vmin=np.percentile(positive, 5),
                vmax=np.max(positive)
            ),
            interpolation="nearest",
            aspect="auto"
        )
    else:
        im = ax.imshow(matrix, cmap='Reds', interpolation='nearest', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Neuron index')
    ax.set_ylabel('Neuron index')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mutual information (bits)')
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def save_mi_outputs(spike_mon: SpikeMonitor,
                    bin_width_ms: float,
                    matrix_output_npz: str,
                    heatmap_output_png: str,
                    title: str) -> np.ndarray:
    """Compute MI matrix, save raw matrix and heatmap, and return matrix."""
    mi_matrix = bin_spikes_and_compute_mi_matrix_corrected(spike_mon=spike_mon,
                                                           bin_width_ms=bin_width_ms,
                                                           n_null=100,
                                                           lag=10,
                                                           method='grass')

    matrix_out = Path(matrix_output_npz)
    matrix_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_out,
        mi_matrix=mi_matrix.astype(np.float32),
        bin_width_ms=np.float32(bin_width_ms),
    )

    save_mi_heatmap(mi_matrix=mi_matrix, 
                    output_path=heatmap_output_png, 
                    lognorm=False,
                    title=title)
    return mi_matrix

