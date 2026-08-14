import argparse
import csv
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.linalg import inv
from scipy import sparse

from mutual_information import bin_spike_events_and_compute_mi_matrix_corrected


def deltacon_similarity_from_adj(A1: np.ndarray, 
                                 A2: np.ndarray, 
                                 epsilon: float = 0.01) -> float:
    """
    Compute DeltaCon similarity between two graphs given by adjacency matrices.
    Parameters:
        A1, A2 : np.ndarray
            Binary (0/1) adjacency matrices of same shape (n x n)
        epsilon : float
            Small constant used in the affinity computation
    Returns:
        similarity : float
            DeltaCon similarity in [0, 1]
    """
    if A1.shape != A2.shape:
        raise ValueError('Adjacency matrices must have the same dimensions')
    if A1.ndim != 2 or A1.shape[0] != A1.shape[1]:
        raise ValueError('Adjacency matrices must be square')

    # DeltaCon is defined for undirected graphs; for this usecase we compare
    # directed/weighted matrices via their undirected support.
    A1 = np.asarray(A1, dtype=np.float64)
    A2 = np.asarray(A2, dtype=np.float64)
    A1 = np.maximum(A1, A1.T)
    A2 = np.maximum(A2, A2.T)
    A1 = np.clip(A1, 0.0, None)
    A2 = np.clip(A2, 0.0, None)
    np.fill_diagonal(A1, 0.0)
    np.fill_diagonal(A2, 0.0)

    n = A1.shape[0]
    I = np.eye(n, dtype=np.float64)

    def affinity_matrix(A: np.ndarray) -> np.ndarray:
        D = np.diag(np.sum(A, axis=1))
        M = I + (epsilon ** 2) * D - epsilon * A
        # Solve M X = I rather than forming inv(M) explicitly.
        S = np.linalg.solve(M, I)
        # Numerical noise can create tiny negative values; clip before sqrt.
        return np.clip(S, 0.0, None)

    S1 = affinity_matrix(A1)
    S2 = affinity_matrix(A2)

    diff = np.sqrt(S1) - np.sqrt(S2)
    d = np.linalg.norm(diff, ord='fro')
    return float(1.0 / (1.0 + d))


def Connector(Q):
    D = nx.to_networkx_graph(Q,create_using=nx.DiGraph())
    Isolate_list=list(nx.isolates(D))
    if len(Isolate_list)>0:
        for i in Isolate_list:
            if i==0:
                Q[i+1,i]=0.0001
            else:
                Q[i-1,i]=0.0001
    del D
    return Q

 
def kNN(A,N,k):
    np.fill_diagonal(A,0)
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
 
    # W.sort(reverse=True)
    B1 = np.zeros((N, N))
    for i in range(N):
        W=sorted(A[i,:],reverse=True)
    #     print( W[k])
        B1[i,:]=np.where(A[i,:] > W[k], 1, 0)
 
    # B=np.multiply(B1,A)
    # print(W[k])
    # print(A[20,1:20])
    # print(B[20,1:20])
 
 
    C1 = np.zeros((N, N))
    for i in range(N):
        W=sorted(A[:,i],reverse=True)
    # print( W[k])
        C1[:,i]=np.where(A[:,i] > W[k], 1, 0)
    # C=np.multiply(C1,A)
    Q1=B1+C1    
    Q2=np.where(Q1 > .9 , 1, 0)
 
    Q=np.multiply(Q2,A)
    # del A
    del B1
    del C1
    del Q1
    del Q2 
    Connector(Q)
 
    return Q

def binarize(A, threshold=0.01):
    A=np.where(A > threshold, 1, 0)
    return A


def load_npz_matrix(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as z:
        if key not in z:
            raise KeyError(f"Key '{key}' not found in {npz_path}")
        return np.asarray(z[key], dtype=np.float64)


def load_scalar(npz_path: Path, key: str) -> float:
    with np.load(npz_path, allow_pickle=False) as z:
        if key not in z:
            raise KeyError(f"Key '{key}' not found in {npz_path}")
        return float(np.asarray(z[key]).item())


def load_spike_events(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load excitatory spike events and neuron count from an internal NPZ file."""
    with np.load(npz_path, allow_pickle=False) as z:
        if 'exc_spike_i' not in z or 'exc_spike_t_ms' not in z:
            raise KeyError(
                f"Spike file {npz_path} must contain 'exc_spike_i' and 'exc_spike_t_ms'."
            )
        spike_i = np.asarray(z['exc_spike_i'], dtype=np.int64)
        spike_t_ms = np.asarray(z['exc_spike_t_ms'], dtype=np.float64)
        if 'n_readout_exc' not in z:
            raise KeyError(f"Spike file {npz_path} must contain 'n_readout_exc'.")
        n_neurons = int(np.asarray(z['n_readout_exc']).item())
    return spike_i, spike_t_ms, n_neurons


def save_mi_matrix_npz(mi_matrix: np.ndarray, output_path: Path, bin_width_ms: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mi_matrix=np.asarray(mi_matrix, dtype=np.float32),
        bin_width_ms=np.float32(bin_width_ms),
    )


def recompute_mi_matrices_from_spikes(
    input_internal_dir: Path,
    output_internal_dir: Path,
    bin_width_ms: float,
    lag_ms: float,
) -> None:
    """Recompute both readout MI matrices from saved spike-event artifacts."""
    multilayer_spikes_path = input_internal_dir / 'multilayer_readout_spikes.npz'
    neuropil_spikes_path = input_internal_dir / 'readout_layer_simulation.npz'

    if not multilayer_spikes_path.exists():
        raise FileNotFoundError(
            f"Missing multilayer spike artifact: {multilayer_spikes_path}"
        )
    if not neuropil_spikes_path.exists():
        raise FileNotFoundError(
            f"Missing neuropil spike artifact: {neuropil_spikes_path}"
        )

    multi_i, multi_t, multi_n = load_spike_events(multilayer_spikes_path)
    neuro_i, neuro_t, neuro_n = load_spike_events(neuropil_spikes_path)

    lag_bins = max(0, int(np.round(float(lag_ms) / float(bin_width_ms))))

    mi_multi = bin_spike_events_and_compute_mi_matrix_corrected(
        spike_i=multi_i,
        spike_t_ms=multi_t,
        n_neurons=multi_n,
        bin_width_ms=bin_width_ms,
        n_null=100,
        lag=lag_bins,
        method='grass',
    )
    mi_neuro = bin_spike_events_and_compute_mi_matrix_corrected(
        spike_i=neuro_i,
        spike_t_ms=neuro_t,
        n_neurons=neuro_n,
        bin_width_ms=bin_width_ms,
        n_null=100,
        lag=lag_bins,
        method='grass',
    )

    save_mi_matrix_npz(
        mi_multi,
        output_internal_dir / 'mi_matrix_multilayer_readout_exc.npz',
        bin_width_ms,
    )
    save_mi_matrix_npz(
        mi_neuro,
        output_internal_dir / 'mi_matrix_neuropil_readout_exc.npz',
        bin_width_ms,
    )

    print(
        'Recomputed MI matrices from saved spikes: '
        f"multilayer={mi_multi.shape}, neuropil={mi_neuro.shape}, "
        f"bin_width_ms={bin_width_ms:.6f}, lag_ms={lag_ms:.6f}"
    )


def load_readout_structural_adj(internal_dir: Path) -> np.ndarray:
    """Extract readout E->E structural block from global adjacency + offsets."""
    adj_npz = internal_dir / 'adjacency_global_sparse.npz'
    offsets_csv = internal_dir / 'adjacency_global_layer_offsets.csv'

    if not adj_npz.exists():
        raise FileNotFoundError(f"Missing structural adjacency file: {adj_npz}")
    if not offsets_csv.exists():
        raise FileNotFoundError(f"Missing layer-offset file: {offsets_csv}")

    with np.load(adj_npz, allow_pickle=False) as z:
        A_global = sparse.csr_matrix(
            (z['data'], z['indices'], z['indptr']),
            shape=tuple(np.asarray(z['shape'], dtype=np.int64)),
        )

    rows = []
    with open(offsets_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError(f"No rows found in offsets file: {offsets_csv}")

    # Readout layer is last layer in offsets table.
    last = rows[-1]
    exc_offset = int(last['exc_offset'])
    n_exc = int(last['n_exc'])
    block = A_global[exc_offset:exc_offset + n_exc, exc_offset:exc_offset + n_exc].toarray()
    return np.asarray(block, dtype=np.float64)


def compute_avg_degree_from_structural(structural_adj: np.ndarray) -> float:
    """Average out-degree from binary directed structural adjacency."""
    S_bin = np.where(structural_adj != 0, 1.0, 0.0)
    np.fill_diagonal(S_bin, 0.0)
    n = S_bin.shape[0]
    edge_count = float(np.sum(S_bin))
    return edge_count / float(n)


def knn_functional_binary(mi_matrix: np.ndarray, k_val: int) -> np.ndarray:
    n = mi_matrix.shape[0]
    k_eff = max(1, min(int(k_val), n - 2))
    func_w = kNN(np.array(mi_matrix, dtype=np.float64, copy=True), n, k_eff)
    func_bin = binarize(func_w, threshold=0.0).astype(np.float64)
    np.fill_diagonal(func_bin, 0.0)
    return func_bin


def randomize_binary_network_preserve_degree(adj_bin: np.ndarray,
                                             n_swaps: int | None = None,
                                             seed: int | None = None) -> np.ndarray:
    """Degree-preserving null model via double-edge swaps on the undirected binarized graph."""
    undirected_adj = np.maximum(adj_bin, adj_bin.T)
    np.fill_diagonal(undirected_adj, 0.0)
    G = nx.from_numpy_array(undirected_adj)

    n_edges = G.number_of_edges()
    if n_swaps is None:
        n_swaps = 10 * n_edges  # standard heuristic for adequate mixing

    nx.double_edge_swap(G, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)

    randomized_adj = nx.to_numpy_array(G, nodelist=range(adj_bin.shape[0]))
    return np.where(randomized_adj != 0, 1.0, 0.0)


def louvain_partition_labels_from_adj(adj_bin: np.ndarray,
                                      resolution: float = 1.0) -> np.ndarray:
    """Run Louvain on an undirected view of adjacency and return integer labels per node."""
    undirected_adj = np.maximum(adj_bin, adj_bin.T)
    G = nx.from_numpy_array(undirected_adj)
    communities = nx.community.louvain_communities(G, weight='weight', resolution=resolution, seed=0)

    labels = np.full(G.number_of_nodes(), -1, dtype=np.int32)
    for comm_idx, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            labels[int(node)] = comm_idx

    if np.any(labels < 0):
        raise RuntimeError('Louvain did not assign all nodes to communities.')
    return labels, len(communities)


def normalized_mutual_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """NMI = 2I(A;B) / (H(A) + H(B)) using natural log."""
    if labels_a.shape != labels_b.shape:
        raise ValueError('Partition label arrays must have identical shape.')

    n = int(labels_a.size)
    if n == 0:
        raise ValueError('Partition label arrays must be non-empty.')

    uniq_a, inv_a = np.unique(labels_a, return_inverse=True)
    uniq_b, inv_b = np.unique(labels_b, return_inverse=True)
    n_a = int(uniq_a.size)
    n_b = int(uniq_b.size)

    contingency = np.zeros((n_a, n_b), dtype=np.float64)
    np.add.at(contingency, (inv_a, inv_b), 1.0)

    p_ab = contingency / float(n)
    p_a = np.sum(p_ab, axis=1)
    p_b = np.sum(p_ab, axis=0)

    nz = p_ab > 0.0
    rows, cols = np.where(nz)
    mi = float(np.sum(p_ab[rows, cols] * np.log(p_ab[rows, cols] / (p_a[rows] * p_b[cols]))))

    nz_a = p_a > 0.0
    nz_b = p_b > 0.0
    h_a = float(-np.sum(p_a[nz_a] * np.log(p_a[nz_a])))
    h_b = float(-np.sum(p_b[nz_b] * np.log(p_b[nz_b])))

    denom = h_a + h_b
    if denom <= 0.0:
        return 1.0 if np.array_equal(labels_a, labels_b) else 0.0
    return float(2.0 * mi / denom)


def append_results_row(
    output_path: Path,
    duration_ms: float,
    r_ee: float,
    mi_bin_size_ms: float,
    summary_signal: str,
    summary_dt_ms: float,
    layer_weight_decay_lambda: float,
    radius_um: float,
    k_used_knn: int,
    louvain_resolution: float,
    n_communities_a: int,
    n_communities_b: int,
    louvain_nmi: float,
    structural_a_functional_a: float,
    structural_b_functional_b: float,
    functional_a_functional_b: float,
) -> None:
    header = [
        'duration_ms',
        'R_ee',
        'mi_bin_size_ms',
        'summary_signal',
        'summary_dt_ms',
        'layer_weight_decay_lambda',
        'radius_um',
        'k_used_knn',
        'louvain_resolution',
        'n_communities_a',
        'n_communities_b',
        'louvain_nmi',
        'structural_a_functional_a',
        'structural_b_functional_b',
        'functional_a_functional_b',
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not output_path.exists()) or (output_path.stat().st_size == 0)
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            f"{duration_ms:.6f}",
            f"{r_ee:.6f}",
            f"{mi_bin_size_ms:.6f}",
            summary_signal,
            f"{summary_dt_ms:.6f}",
            f"{layer_weight_decay_lambda:.6f}",
            f"{radius_um:.6f}",
            str(int(k_used_knn)),
            f"{louvain_resolution:.6f}",
            str(int(n_communities_a)),
            str(int(n_communities_b)),
            f"{louvain_nmi:.6f}",
            f"{structural_a_functional_a:.6f}",
            f"{structural_b_functional_b:.6f}",
            f"{functional_a_functional_b:.6f}",
        ])


def run_analysis(
    input_internal_dir: Path,
    output_internal_dir: Path,
    output_path: Path,
    r_ee: float,
    duration_ms: float,
    summary_signal: str,
    summary_dt_ms: float,
    layer_weight_decay_lambda: float,
    radius_um: float,
    recompute_mi_from_spikes: bool,
    mi_bin_width_ms: float,
    mi_lag_ms: float,
    louvain_resolution: float,
) -> None:
    mi_a_path = output_internal_dir / 'mi_matrix_multilayer_readout_exc.npz'
    mi_b_path = output_internal_dir / 'mi_matrix_neuropil_readout_exc.npz'

    if recompute_mi_from_spikes or (not mi_a_path.exists()) or (not mi_b_path.exists()):
        recompute_mi_matrices_from_spikes(
            input_internal_dir=input_internal_dir,
            output_internal_dir=output_internal_dir,
            bin_width_ms=mi_bin_width_ms,
            lag_ms=mi_lag_ms,
        )

    mi_a = load_npz_matrix(mi_a_path, 'mi_matrix')
    mi_b = load_npz_matrix(mi_b_path, 'mi_matrix')
    if mi_a.shape != mi_b.shape:
        raise ValueError(f"MI matrices must have same shape, got {mi_a.shape} and {mi_b.shape}")

    mi_bin_a = load_scalar(mi_a_path, 'bin_width_ms')
    mi_bin_b = load_scalar(mi_b_path, 'bin_width_ms')
    if abs(mi_bin_a - mi_bin_b) > 1e-9:
        raise ValueError(f"MI bin widths differ: {mi_bin_a} vs {mi_bin_b}")

    structural_adj = load_readout_structural_adj(input_internal_dir)
    if structural_adj.shape != mi_a.shape:
        raise ValueError(
            'Structural readout adjacency shape does not match MI matrix shape: '
            f"{structural_adj.shape} vs {mi_a.shape}"
        )

    avg_degree = compute_avg_degree_from_structural(structural_adj)
    k_val = int(round(avg_degree))
    k_eff = max(1, min(k_val, structural_adj.shape[0] - 2))

    structural_bin = np.where(structural_adj != 0, 1.0, 0.0)
    np.fill_diagonal(structural_bin, 0.0)
    functional_a_bin = knn_functional_binary(mi_a, k_eff)
    functional_b_bin = knn_functional_binary(mi_b, k_eff)

    louvain_labels_a, n_communities_a = louvain_partition_labels_from_adj(functional_a_bin, resolution=louvain_resolution)
    louvain_labels_b, n_communities_b = louvain_partition_labels_from_adj(functional_b_bin, resolution=louvain_resolution)
    louvain_nmi = normalized_mutual_information(louvain_labels_a, louvain_labels_b)

    sim_struct_a_func_a = float(deltacon_similarity_from_adj(structural_bin, functional_a_bin))
    sim_struct_b_func_b = float(deltacon_similarity_from_adj(structural_bin, functional_b_bin))
    sim_func_a_func_b = float(deltacon_similarity_from_adj(functional_a_bin, functional_b_bin))

    append_results_row(
        output_path=output_path,
        duration_ms=float(duration_ms),
        r_ee=float(r_ee),
        mi_bin_size_ms=float(mi_bin_a),
        summary_signal=summary_signal,
        summary_dt_ms=float(summary_dt_ms),
        layer_weight_decay_lambda=float(layer_weight_decay_lambda),
        radius_um=float(radius_um),
        k_used_knn=int(k_eff),
        louvain_resolution=float(louvain_resolution),
        n_communities_a=int(n_communities_a),
        n_communities_b=int(n_communities_b),
        louvain_nmi=float(louvain_nmi),
        structural_a_functional_a=sim_struct_a_func_a,
        structural_b_functional_b=sim_struct_b_func_b,
        functional_a_functional_b=sim_func_a_func_b,
    )

    print(f"Readout average degree (structural): {avg_degree:.6f}")
    print(f"k used for kNN functional construction: {k_eff}")
    print(f"DeltaCon structural_a_functional_a: {sim_struct_a_func_a:.6f}")
    print(f"DeltaCon structural_b_functional_b: {sim_struct_b_func_b:.6f}")
    print(f"DeltaCon functional_a_functional_b: {sim_func_a_func_b:.6f}")
    print(f"Louvain partition NMI (functional_a vs functional_b): {louvain_nmi:.6f}")
    print(f"Appended results row to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare structural and functional readout networks.')
    parser.add_argument(
        '--input-internal-dir',
        default='output/internal',
        help='Directory containing reusable simulation outputs and structural artifacts.',
    )
    parser.add_argument(
        '--output-internal-dir',
        default='output/internal',
        help='Directory where derived MI matrices for this analysis run are written/read.',
    )
    parser.add_argument('--output-file', default='output/public/network_compare_results.csv', help='Append-only results file path.')
    parser.add_argument('--r-ee', type=float, default=2.0, help='R_ee value to record in results.')
    parser.add_argument('--duration-ms', type=float, default=2000.0, help='Duration to record in results.')
    parser.add_argument('--summary-signal', choices=['v', 'i_syn', 'g_e'], default='v', help='Summary signal metadata value from multilayer run.')
    parser.add_argument('--summary-dt-ms', type=float, default=0.1, help='Summary timestep metadata value from multilayer run.')
    parser.add_argument('--layer-weight-decay-lambda', type=float, default=1.0, help='Layer weighting decay metadata value from neuropil run.')
    parser.add_argument('--radius-um', type=float, default=25.0, help='Neighborhood radius metadata value from neuropil run.')
    parser.add_argument(
        '--recompute-mi-from-spikes',
        action='store_true',
        help='Recompute MI matrices from saved readout spike-event artifacts before analysis.',
    )
    parser.add_argument(
        '--mi-bin-width-ms',
        type=float,
        default=10.0,
        help='Bin width (ms) used only when --recompute-mi-from-spikes is enabled or MI files are missing.',
    )
    parser.add_argument(
        '--mi-lag-ms',
        type=float,
        default=10.0,
        help='Lag window (ms) used when recomputing MI from saved spikes.',
    )
    parser.add_argument(
        '--louvain-resolution',
        type=float,
        default=1.0,
        help='Resolution parameter for Louvain community detection in network_analysis.py.',
    )
    args = parser.parse_args()

    if args.duration_ms <= 0.0:
        raise ValueError('--duration-ms must be positive.')
    if args.summary_dt_ms <= 0.0:
        raise ValueError('--summary-dt-ms must be positive.')
    if args.mi_bin_width_ms <= 0.0:
        raise ValueError('--mi-bin-width-ms must be positive.')
    if args.mi_lag_ms < 0.0:
        raise ValueError('--mi-lag-ms must be non-negative.')

    run_analysis(
        input_internal_dir=Path(args.input_internal_dir),
        output_internal_dir=Path(args.output_internal_dir),
        output_path=Path(args.output_file),
        r_ee=float(args.r_ee),
        duration_ms=float(args.duration_ms),
        summary_signal=args.summary_signal,
        summary_dt_ms=float(args.summary_dt_ms),
        layer_weight_decay_lambda=float(args.layer_weight_decay_lambda),
        radius_um=float(args.radius_um),
        recompute_mi_from_spikes=bool(args.recompute_mi_from_spikes),
        mi_bin_width_ms=float(args.mi_bin_width_ms),
        mi_lag_ms=float(args.mi_lag_ms),
        louvain_resolution=float(args.louvain_resolution),
    )


if __name__ == '__main__':
    main()
