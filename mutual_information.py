from brian2 import SpikeMonitor, ms
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def compute_mi_binary(x: np.ndarray, y: np.ndarray) -> float:
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

    return float(mi)

def compute_mi_matrix(spike_trains: np.ndarray) -> np.ndarray:
    """
    Compute mutual information matrix for a set of spike trains
    """
    n_neurons = spike_trains.shape[0]
    mi_matrix = np.zeros((n_neurons, n_neurons), dtype=float)

    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            mi = compute_mi_binary(spike_trains[i], spike_trains[j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def construct_mi_matrix(spike_mon: SpikeMonitor,
                        bin_width_ms: float) -> np.ndarray:
    """Bin spikes, binarize occupancy, then compute pairwise MI matrix."""
    spike_counts = bin_spikes(spike_mon=spike_mon, bin_width_ms=bin_width_ms)
    spike_binary = (spike_counts > 0).astype(np.int8)
    return compute_mi_matrix(spike_binary)


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
    mi_matrix = construct_mi_matrix(spike_mon=spike_mon, bin_width_ms=bin_width_ms)

    matrix_out = Path(matrix_output_npz)
    matrix_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_out,
        mi_matrix=mi_matrix.astype(np.float32),
        bin_width_ms=np.float32(bin_width_ms),
    )

    save_mi_heatmap(mi_matrix=mi_matrix, 
                    output_path=heatmap_output_png, 
                    lognorm=True,
                    title=title)
    return mi_matrix
