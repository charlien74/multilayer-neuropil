import numpy as np
import matplotlib.pyplot as plt
from brian2 import Hz
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs


def _layer_index_info(layers):
    """Return per-layer global index offsets for E and I populations."""
    layer_info = []
    offset = 0
    for layer in layers:
        n_exc = int(len(layer['exc_neurons']))
        n_inh = int(len(layer['inh_neurons']))
        info = {
            'exc_offset': offset,
            'inh_offset': offset + n_exc,
            'n_exc': n_exc,
            'n_inh': n_inh,
        }
        layer_info.append(info)
        offset += n_exc + n_inh
    return layer_info, offset


def _append_synapses(rows, cols, data, syn, pre_offset, post_offset):
    pre = np.asarray(syn.i[:], dtype=np.int64) + int(pre_offset)
    post = np.asarray(syn.j[:], dtype=np.int64) + int(post_offset)
    w = np.asarray(syn.w_syn[:] / Hz, dtype=np.float64)
    if pre.size == 0:
        return
    rows.append(pre)
    cols.append(post)
    data.append(w)


def extract_global_weighted_adjacency(layers, interlayer_synapses):
    """Build a sparse weighted adjacency matrix from Brian2 Synapses objects."""
    layer_info, n_total = _layer_index_info(layers)
    rows, cols, data = [], [], []

    for layer_i, layer in enumerate(layers):
        info = layer_info[layer_i]
        _append_synapses(rows, cols, data, layer['syn_ee'], info['exc_offset'], info['exc_offset'])
        _append_synapses(rows, cols, data, layer['syn_ii'], info['inh_offset'], info['inh_offset'])
        _append_synapses(rows, cols, data, layer['syn_ei'], info['inh_offset'], info['exc_offset'])
        _append_synapses(rows, cols, data, layer['syn_ie'], info['exc_offset'], info['inh_offset'])

    for layer_i, inter_syn in enumerate(interlayer_synapses):
        pre_info = layer_info[layer_i]
        post_info = layer_info[layer_i + 1]
        _append_synapses(rows, cols, data, inter_syn['syn_ee'], pre_info['exc_offset'], post_info['exc_offset'])
        _append_synapses(rows, cols, data, inter_syn['syn_ei'], pre_info['exc_offset'], post_info['inh_offset'])
        _append_synapses(rows, cols, data, inter_syn['syn_ie'], pre_info['inh_offset'], post_info['exc_offset'])
        _append_synapses(rows, cols, data, inter_syn['syn_ii'], pre_info['inh_offset'], post_info['inh_offset'])

    if not rows:
        adjacency = coo_matrix((n_total, n_total), dtype=np.float64).tocsr()
    else:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        dat = np.concatenate(data)
        adjacency = coo_matrix((dat, (row, col)), shape=(n_total, n_total), dtype=np.float64).tocsr()

    return adjacency, layer_info


def extract_layer_ee_adjacency(layer):
    """Return dense E-E weighted adjacency for a single layer."""
    n_exc = int(len(layer['exc_neurons']))
    adj = np.zeros((n_exc, n_exc), dtype=np.float64)
    pre = np.asarray(layer['syn_ee'].i[:], dtype=np.int64)
    post = np.asarray(layer['syn_ee'].j[:], dtype=np.int64)
    w = np.asarray(layer['syn_ee'].w_syn[:] / Hz, dtype=np.float64)
    if pre.size > 0:
        adj[pre, post] = w
    return adj


def compute_layer_ee_eigenvalues(layers):
    """Compute full E-E eigenspectrum per layer."""
    eigenvalues = []
    for layer in layers:
        adj = extract_layer_ee_adjacency(layer)
        eigenvalues.append(np.linalg.eigvals(adj))
    return eigenvalues


def compute_global_dominant_eigenvalues(adjacency, k=120):
    """Compute dominant eigenvalues for a large sparse global adjacency matrix."""
    n = adjacency.shape[0]
    if n < 3:
        dense = adjacency.toarray()
        return np.linalg.eigvals(dense)

    k_eff = int(min(k, n - 2))
    if k_eff < 1:
        dense = adjacency.toarray()
        return np.linalg.eigvals(dense)

    vals = eigs(adjacency, k=k_eff, which='LR', return_eigenvectors=False)
    return vals


def plot_complex_spectrum(eigenvalues, output_path, title):
    """Scatter plot of eigenvalues in the complex plane."""
    vals = np.asarray(eigenvalues)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(vals.real, vals.imag, s=18, alpha=0.75, color='tab:blue', edgecolors='none')
    ax.axhline(0.0, color='0.3', linewidth=0.8)
    ax.axvline(0.0, color='0.3', linewidth=0.8)
    ax.set_xlabel('Real(lambda)')
    ax.set_ylabel('Imag(lambda)')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_layer_ee_spectra(layer_eigs, output_path):
    """Plot each layer's E-E eigenspectrum in separate panels."""
    n_layers = len(layer_eigs)
    fig, axes = plt.subplots(n_layers, 1, figsize=(7, 4 * n_layers), squeeze=False)
    for layer_i in range(n_layers):
        vals = np.asarray(layer_eigs[layer_i])
        ax = axes[layer_i, 0]
        ax.scatter(vals.real, vals.imag, s=10, alpha=0.7, color='tab:orange', edgecolors='none')
        ax.axhline(0.0, color='0.35', linewidth=0.7)
        ax.axvline(0.0, color='0.35', linewidth=0.7)
        ax.set_xlabel('Real(lambda)')
        ax.set_ylabel('Imag(lambda)')
        ax.set_title(f'Layer {layer_i} E-E eigenvalues')
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
