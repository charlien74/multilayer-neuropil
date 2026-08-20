from brian2 import SpikeMonitor, ms
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import math
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
    """
    Build a squared-exponential spatial kernel matrix.

    K[i, j] gives the contribution of neuron j to the activity field
    evaluated at neuron i's spatial location.

    The kernel is not row-normalized: densely populated / highly active
    regions therefore naturally produce larger field amplitudes.
    """
    if spatial_sigma_um <= 0:
        raise ValueError("spatial_sigma_um must be > 0.")

    positions = np.asarray(positions_um, dtype=np.float32)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_um must have shape (N, 2).")

    dx = positions[:, 0][:, None] - positions[:, 0][None, :]
    dy = positions[:, 1][:, None] - positions[:, 1][None, :]
    sqdist = dx * dx + dy * dy

    sigma2 = float(spatial_sigma_um ** 2)

    k = np.exp(
        -sqdist / (2.0 * sigma2)
    ).astype(np.float32)

    return k


def apply_temporal_kernel(window_counts: np.ndarray,
                          dt_ms: float,
                          tau_ms: float,
                          normalize: bool = True) -> np.ndarray:
    """
    Apply causal exponential smoothing to a full recording.

    Implements the discrete analogue of

        k_t(t, t_s)
            = H(t - t_s) exp(-(t - t_s) / tau)

    or, if normalize=True,

        k_t(t, t_s)
            = (1 / tau) H(t - t_s) exp(-(t - t_s) / tau).

    Parameters
    ----------
    counts : (N, T) array
        Spike counts for N neurons across T time bins.
    dt_ms : float
        Width of each time bin in ms.
    tau_ms : float
        Temporal decay constant in ms.
    normalize : bool
        If True, multiply each spike contribution by 1 / tau_ms.
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0.")
    if tau_ms <= 0:
        raise ValueError("tau_ms must be > 0.")

    counts = np.asarray(window_counts, dtype=np.float32)

    if counts.ndim != 2:
        raise ValueError("counts must have shape (N, T).")
    if counts.shape[1] == 0:
        raise ValueError("counts must contain at least one time bin.")

    smoothed = np.zeros_like(counts)

    alpha = float(np.exp(-dt_ms / tau_ms))
    scale = 1.0 / tau_ms if normalize else 1.0

    smoothed[:, 0] = scale * counts[:, 0]

    for t in range(1, counts.shape[1]):
        smoothed[:, t] = (
            alpha * smoothed[:, t - 1]
            + scale * counts[:, t]
        )

    return smoothed


def build_spatiotemporal_window_features(spike_i: np.ndarray,
                                         spike_t_ms: np.ndarray,
                                         positions_um: np.ndarray,
                                         n_neurons: int,
                                         window_ms: float,
                                         dt_ms: float,
                                         tau_ms: float,
                                         spatial_sigma_um: float,
                                         t_start_ms: float | None = None,
                                         t_stop_ms: float | None = None) -> dict:
    """Build flattened FA(x,t)-like features per window from a SpikeMonitor."""
    spike_i = np.asarray(spike_i, dtype=np.int64)
    spike_t_ms = np.asarray(spike_t_ms, dtype=np.float64)

    if spike_i.shape != spike_t_ms.shape:
        raise ValueError("spike_i and spike_t_ms must have the same shape.")

    spike_time_list = [
        spike_t_ms[spike_i == neuron_idx]
        for neuron_idx in range(n_neurons)
    ]

    # Bin population: shape (n_neurons, n_time_bins)
    counts = bin_population(
        spike_time_list,
        t_start=t_start_ms,
        t_stop=t_stop_ms,
        dt=dt_ms,
        clip=1,
    )

    k_spatial = build_spatial_kernel(positions_um, spatial_sigma_um=spatial_sigma_um)

    temporal = apply_temporal_kernel(
        counts,
        dt_ms=dt_ms,
        tau_ms=tau_ms,
    )

    spatial_temporal = k_spatial @ temporal

    windows = split_count_matrix_into_windows(
        spatial_temporal,
        window_ms=window_ms,
        dt_ms=dt_ms,
    )

    features = np.asarray(
        [w.reshape(-1) for w in windows],
        dtype=np.float32,
    )

    return {
        'features': features,
        'positions_um': positions_um,
        'dt_ms': float(dt_ms),
        'window_ms': float(window_ms),
        't_start_ms': float(t_start_ms),
        't_stop_ms': float(t_stop_ms),
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
    """Return k-nearest-neighbor index sets per sample (including self)."""
    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError('distance_matrix must be square.')
    if h <= 0:
        raise ValueError('h must be > 0.')

    n = d.shape[0]
    neighbors = []
    for i in range(n):
        row = d[i].copy()
        row[i] = 0.0
        idx = np.argpartition(row, h)[:h]
        neighbors.append(set(int(v) for v in idx))
    return neighbors


def estimate_distance_based_mi_from_distance_matrices(distance_a: np.ndarray,
                                                      distance_b: np.ndarray,
                                                      h: int) -> dict:
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

    if not h < n:
        raise ValueError('Need h less than the number of windows for estimator to make sense.')

    neigh_a = k_nearest_neighbor_sets(da, h=h)
    neigh_b = k_nearest_neighbor_sets(db, h=h)

    intersections = np.zeros(n, dtype=np.int32)
    for i in range(n):
        intersections[i] = len(neigh_a[i].intersection(neigh_b[i]))

    if np.any(intersections == 0):
        raise ValueError(
            "Zero nearest-neighbor intersections encountered. "
            "Increase h or use the bias-corrected estimator."
        )

    intersections = intersections.astype(np.float64)
    terms = np.log2((n * intersections) / float(h * h))
    mi_bits = float(np.mean(terms))

    mi_null = 0
    for r in range(1, h + 1):
        prob = (math.comb(h - 1, r - 1) * math.comb(n - h, h - r)) / math.comb(n - 1, h - 1)
        if prob > 0:
            mi_null += prob * np.log2((n * r) / float(h * h))

    return {
        'mi_bits': mi_bits,
        'mi_null_bits': mi_null,
        'mi_corrected_bits': mi_bits - mi_null,
        'intersection_counts': intersections,
        'h': h,
        'n_windows': int(n),
    }


def estimate_distance_based_mi(spike_i_a: np.ndarray,
                               spike_t_ms_a: np.ndarray,
                               positions_um_a: np.ndarray,
                               spike_i_b: np.ndarray,
                               spike_t_ms_b: np.ndarray,
                               positions_um_b: np.ndarray,
                               n_neurons: int,
                               window_ms: float,
                               dt_ms: float,
                               tau_ms: float,
                               spatial_sigma_um: float,
                               h: int,
                               t_start_ms: float | None = None,
                               t_stop_ms: float | None = None) -> dict:
    """End-to-end distance-based MI estimate between two SpikeMonitor recordings."""
    if t_start_ms is None or t_stop_ms is None:
        ta = np.asarray(spike_t_ms_a, dtype=float)
        tb = np.asarray(spike_t_ms_b, dtype=float)
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
        spike_i_a,
        spike_t_ms_a,
        positions_um_a,
        n_neurons,
        window_ms=window_ms,
        dt_ms=dt_ms,
        tau_ms=tau_ms,
        spatial_sigma_um=spatial_sigma_um,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
    )
    feats_b = build_spatiotemporal_window_features(
        spike_i_b,
        spike_t_ms_b,
        positions_um_b,
        n_neurons,
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
    return estimate_distance_based_mi_from_distance_matrices(
        d_a,
        d_b,
        h=h
    )


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

# def compute_mi_matrix(spike_trains: np.ndarray, entropy_normalize: bool = False) -> np.ndarray:
#     """
#     Compute mutual information matrix for a set of spike trains
#     """
#     n_neurons = spike_trains.shape[0]
#     mi_matrix = np.zeros((n_neurons, n_neurons), dtype=float)

#     for i in range(n_neurons):
#         for j in range(i + 1, n_neurons):
#             mi = compute_mi_binary(spike_trains[i], 
#                                    spike_trains[j], 
#                                    entropy_normalize=entropy_normalize)
#             mi_matrix[i, j] = mi
#             mi_matrix[j, i] = mi

#     return mi_matrix


# def bin_spikes_and_compute_mi_matrix(spike_mon: SpikeMonitor,
#                                      bin_width_ms: float,
#                                      entropy_normalize: bool = False) -> np.ndarray:
#     """Bin spikes, binarize occupancy, then compute pairwise MI matrix."""
#     spike_counts = bin_spikes(spike_mon=spike_mon, bin_width_ms=bin_width_ms)
#     spike_binary = (spike_counts > 0).astype(np.int8)
#     return compute_mi_matrix(spike_binary, entropy_normalize=entropy_normalize)


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


def bin_spike_events_and_compute_mi_matrix_corrected(
    spike_i: np.ndarray,
    spike_t_ms: np.ndarray,
    n_neurons: int,
    bin_width_ms: float,
    n_null: int = 100,
    lag: int = 10,
    method: str = 'plugin',
) -> np.ndarray:
    """Compute corrected MI from saved spike-event arrays via existing SpikeMonitor path."""

    class _SpikeEventsAdapter:
        def __init__(self, i_arr: np.ndarray, t_arr_ms: np.ndarray, n: int):
            i_arr = np.asarray(i_arr, dtype=np.int64)
            t_arr_ms = np.asarray(t_arr_ms, dtype=np.float64)
            self.t = t_arr_ms * ms
            self._spike_train_dict = {
                idx: t_arr_ms[i_arr == idx] * ms
                for idx in range(int(n))
            }

        def spike_trains(self):
            return self._spike_train_dict

    adapter = _SpikeEventsAdapter(spike_i, spike_t_ms, int(n_neurons))
    return bin_spikes_and_compute_mi_matrix_corrected(
        spike_mon=adapter,
        bin_width_ms=bin_width_ms,
        n_null=n_null,
        lag=lag,
        method=method,
    )


def spatial_region_ids(positions_um: np.ndarray,
                       n_regions: int,
                       spatial_radius_um: float) -> np.ndarray:
    """Assign positions to an equal-area square grid centered at the origin."""
    positions = np.asarray(positions_um, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError('positions_um must have shape (n_neurons, 2).')
    if n_regions <= 0 or int(np.sqrt(n_regions)) ** 2 != n_regions:
        raise ValueError('n_regions must be a positive perfect square.')
    if spatial_radius_um <= 0.0:
        raise ValueError('spatial_radius_um must be positive.')

    side = int(np.sqrt(n_regions))
    normalized = (positions + float(spatial_radius_um)) / (2.0 * float(spatial_radius_um))
    normalized = np.clip(normalized, 0.0, 1.0)
    columns = np.minimum((normalized[:, 0] * side).astype(int), side - 1)
    rows = np.minimum((normalized[:, 1] * side).astype(int), side - 1)
    return rows * side + columns


def save_spatial_region_mi_outputs(
    spike_i: np.ndarray,
    spike_t_ms: np.ndarray,
    positions_um: np.ndarray,
    n_regions: int,
    spatial_radius_um: float,
    bin_width_ms: float,
    matrix_output_npz: str,
    heatmap_output_png: str,
    title: str,
    n_null: int = 100,
    lag: int = 10,
    method: str = 'grass',
    clip: int = 255,
    t_start_ms: float | None = None,
    t_stop_ms: float | None = None,
) -> np.ndarray:
    """Compute and save MI between equal-area spatial regions from spike events."""
    spike_i = np.asarray(spike_i, dtype=np.int64)
    spike_t_ms = np.asarray(spike_t_ms, dtype=np.float64)
    positions_um = np.asarray(positions_um, dtype=np.float64)

    if spike_i.shape != spike_t_ms.shape:
        raise ValueError('spike_i and spike_t_ms must have the same shape.')
    if positions_um.ndim != 2 or positions_um.shape[1] != 2:
        raise ValueError('positions_um must have shape (n_neurons, 2).')
    if np.any(spike_i < 0) or np.any(spike_i >= positions_um.shape[0]):
        raise ValueError('spike_i contains an index outside positions_um.')
    if n_regions <= 0 or int(np.sqrt(n_regions)) ** 2 != n_regions:
        raise ValueError('n_regions must be a positive perfect square.')
    if spatial_radius_um <= 0:
        raise ValueError('spatial_radius_um must be positive.')
    if clip <= 0 or clip > 255:
        raise ValueError('clip must be between 1 and 255.')

    region_ids = spatial_region_ids(
        positions_um,
        n_regions=n_regions,
        spatial_radius_um=spatial_radius_um,
    )

    region_spike_times = [
        spike_t_ms[np.isin(spike_i, np.flatnonzero(region_ids == region))].copy()
        for region in range(n_regions)
    ]

    if t_start_ms is None:
        t_start_ms = float(spike_t_ms.min()) if spike_t_ms.size else 0.0
    if t_stop_ms is None:
        t_stop_ms = float(spike_t_ms.max() + bin_width_ms) if spike_t_ms.size else float(t_start_ms + bin_width_ms)

    regional_states = bin_population(
        region_spike_times,
        t_start=t_start_ms,
        t_stop=t_stop_ms,
        dt=bin_width_ms,
        clip=clip,
    )
    result = mi_pairs(
        regional_states,
        n_null=n_null,
        lag=lag,
    )
    method_index = tuple(result['methods']).index(method)
    mi_matrix = np.zeros((n_regions, n_regions), dtype=float)
    pairs = np.asarray(result['pairs'], dtype=np.int64)
    mi_values = np.asarray(result['mi_corrected'][:, method_index], dtype=float)
    mi_matrix[pairs[:, 0], pairs[:, 1]] = mi_values
    mi_matrix[pairs[:, 1], pairs[:, 0]] = mi_values

    matrix_out = Path(matrix_output_npz)
    matrix_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_out,
        mi_matrix=mi_matrix.astype(np.float32),
        bin_width_ms=np.float32(bin_width_ms),
        n_regions=np.int32(n_regions),
        clip=np.int32(clip),
    )
    save_mi_heatmap(
        mi_matrix=mi_matrix,
        output_path=heatmap_output_png,
        lognorm=False,
        title=title,
    )
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

