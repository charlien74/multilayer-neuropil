from brian2 import SpikeMonitor, ms
import numpy as np

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
