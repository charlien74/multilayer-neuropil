from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from pathlib import Path
from model_util import *

start_scope()
seed(RANDOM_SEED)

weight_decay_l = 50 * um
inh_weight_decay_l = 100 * um
w_ee_base = 4 * 0.0156 * kHz
w_ei_base = -0.0297 * kHz
w_ie_base = 0.0074 * kHz
w_ii_base = -0.0297 * kHz

p_ee_interlayer = 0.1
p_ei_interlayer = 0.1
p_ie_interlayer = 0.1
p_ii_interlayer = 0.1

eqs_exc_multilayer = """
dv/dt = (mu - v) / tau_m + g_e + g_i : 1 (unless refractory)
dg_e/dt = -g_e / tau_e : Hz
dg_i/dt = -g_i / tau_i : Hz
mu : 1
tau_m : second (constant)
column_id : integer (constant)
x: meter
y: meter
"""

eqs_inh_multilayer = """
dv/dt = (mu - v) / tau_m + g_e + g_i : 1 (unless refractory)
dg_e/dt = -g_e / tau_e : Hz
dg_i/dt = -g_i / tau_i : Hz
mu : 1
tau_m : second (constant)
column_id : integer (constant)
x: meter
y: meter
"""

N_layers = 5
uniform_layer_start = N_layers - 5

p_avg=0.01


def assign_nearest_centroid_ids(positions_um, centroids):
    """Assign each neuron to the closest centroid in Euclidean distance."""
    centroids_um = np.array([[float(cx / um), float(cy / um)] for cx, cy in centroids], dtype=float)
    diffs = positions_um[:, None, :] - centroids_um[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    return np.argmin(dist2, axis=1).astype(int)

R_ee = 1.0
interlayer_decay_l = 30 * um
inhibitory_sigma = R / 2
num_exc_per_layer = N_exc_c * 5  # Number of excitatory neurons per layer
uniform_radius = (R + 2 * sigma_c) / um  # Ensure neurons are within a reasonable distance from the center

layers = []
for layer_i in range(N_layers):
    centroids, neuron_locations, positions_um, cluster_ids = generate_pentacle_layout(
        assembly_radius=sigma_c, pentacle_radius=R, n_clusters=5, neurons_per_cluster=N_exc_c) 
    inh_neuron_locations, inh_positions_um, inh_cluster_ids = generate_uniform_layout(radius=uniform_radius, n_neurons=N_inh)

    n_clusters = np.bincount(cluster_ids)
    largest_community_size = int(n_clusters.max()) if n_clusters.size > 0 else num_exc_per_layer
    p_ee_in, p_ee_out = get_p_connection_in_out(
        p_ee_avg=p_avg,
        R_ee=R_ee,
        N_excitatory=num_exc_per_layer,
        cluster_size=largest_community_size,
    )
    ## Remove assemblies from top two layers (readout stack).
    if layer_i >= uniform_layer_start:
        neuron_locations, positions_um, cluster_ids = generate_uniform_layout(radius=uniform_radius, n_neurons=1600)
        inh_neuron_locations, inh_positions_um, _ = generate_uniform_layout(radius=uniform_radius, n_neurons=N_inh)
        cluster_ids = assign_nearest_centroid_ids(positions_um, centroids)
        num_exc_per_layer = len(neuron_locations)  # Update the number of excitatory neurons for the top layer
        p_ee_in = p_avg
        p_ee_out = p_avg
    layer_exc_neurons = NeuronGroup(
        num_exc_per_layer,
        eqs_exc_multilayer,
        threshold='v > v_th',
        reset='v = v_reset',
        refractory=refractory,
        method='euler'
    )
    layer_inh_neurons = inh_neurons = NeuronGroup(
        N_inh,
        eqs_inh_multilayer,
        threshold='v > v_th',
        reset='v = v_reset',
        refractory=refractory,
        method='euler'
    )

    layer_exc_neurons.tau_m = tau_m_e
    layer_exc_neurons.mu = "1.1 + 0.1*rand()"
    layer_exc_neurons.v = "rand()"
    layer_exc_neurons.g_e = 0 * Hz
    layer_exc_neurons.g_i = 0 * Hz

    layer_inh_neurons.tau_m = tau_m_i
    layer_inh_neurons.mu = "1 + 0.05*rand()"
    layer_inh_neurons.v = "rand()"
    layer_inh_neurons.g_e = 0 * Hz
    layer_inh_neurons.g_i = 0 * Hz

    x_coords_um = positions_um[:, 0]
    y_coords_um = positions_um[:, 1]
    layer_exc_neurons.x = x_coords_um * um
    layer_exc_neurons.y = y_coords_um * um
    layer_exc_neurons.column_id = cluster_ids

    inh_x_coords_um = inh_positions_um[:, 0]
    inh_y_coords_um = inh_positions_um[:, 1]
    layer_inh_neurons.x = inh_x_coords_um * um
    layer_inh_neurons.y = inh_y_coords_um * um
    layer_inh_neurons.column_id = inh_cluster_ids

    # E-E probabilities depend only on community membership; weights decay with distance.
    syn_ee = Synapses(layer_exc_neurons, layer_exc_neurons, model="w_syn : Hz", on_pre="g_e_post += w_syn")
    syn_ee.connect(
        condition='i != j',
        p='p_ee_out + (p_ee_in - p_ee_out) * int(column_id_pre == column_id_post)'
    )
    syn_ee.w_syn = 'w_ee_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / weight_decay_l)'

    syn_ii = Synapses(layer_inh_neurons, layer_inh_neurons, model="w_syn : Hz", on_pre="g_i_post += w_syn")
    syn_ii.connect(condition="i != j", p=0.5)
    syn_ii.w_syn = 'w_ii_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / inh_weight_decay_l)'

    syn_ei = Synapses(layer_inh_neurons, layer_exc_neurons, model="w_syn : Hz", on_pre="g_i_post += w_syn")
    syn_ei.connect(p=0.5)
    syn_ei.w_syn = 'w_ei_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / inh_weight_decay_l)'

    syn_ie = Synapses(layer_exc_neurons, layer_inh_neurons, model="w_syn : Hz", on_pre="g_e_post += w_syn")
    syn_ie.connect(p=0.5)
    syn_ie.w_syn = 'w_ie_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / inh_weight_decay_l)'

    layers.append({
        "exc_neurons": layer_exc_neurons,
        "inh_neurons": layer_inh_neurons,
        "syn_ee": syn_ee,
        "syn_ii": syn_ii,
        "syn_ei": syn_ei,
        "syn_ie": syn_ie,
        "positions_um": positions_um,
        "centroids": centroids if layer_i < uniform_layer_start else None,
        "cluster_ids": cluster_ids,
        "p_ee_in": p_ee_in,
        "p_ee_out": p_ee_out,
        })
    
# Establish connections between layers based on distance and decay function
interlayer_synapses = []
for layer_i in range(N_layers - 1):
    pre_layer = layers[layer_i]
    post_layer = layers[layer_i + 1]
    pre_exc = pre_layer['exc_neurons']
    post_exc = post_layer['exc_neurons']
    pre_inh = pre_layer['inh_neurons']
    post_inh = post_layer['inh_neurons']

    # Maybe in the future max_p and decay_l can be different for excitatory vs inhibitory connections?
    syn_ee_inter = Synapses(pre_exc, post_exc, model="w_syn : Hz", on_pre="g_e_post += w_syn")
    syn_ee_inter.connect(condition="True", p='p_ee_interlayer')
    syn_ee_inter.w_syn = 'w_ee_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / weight_decay_l)'
    syn_ei_inter = Synapses(pre_exc, post_inh, model="w_syn : Hz", on_pre="g_e_post += w_syn")
    syn_ei_inter.connect(condition="True", p='p_ei_interlayer')
    syn_ei_inter.w_syn = 'w_ie_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / weight_decay_l)'
    syn_ie_inter = Synapses(pre_inh, post_exc, model="w_syn : Hz", on_pre="g_i_post += w_syn")
    syn_ie_inter.connect(condition="True", p='p_ie_interlayer')
    syn_ie_inter.w_syn = 'w_ei_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / weight_decay_l)'
    syn_ii_inter = Synapses(pre_inh, post_inh, model="w_syn : Hz", on_pre="g_i_post += w_syn")
    syn_ii_inter.connect(condition="True", p='p_ii_interlayer')
    syn_ii_inter.w_syn = 'w_ii_base * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / weight_decay_l)'
    interlayer_synapses.append({'syn_ee': syn_ee_inter,
                                'syn_ei': syn_ei_inter,
                                'syn_ie': syn_ie_inter,
                                'syn_ii': syn_ii_inter})
    
spike_mon_exc = [SpikeMonitor(layer['exc_neurons']) for layer in layers]
spike_mon_inh = [SpikeMonitor(layer['inh_neurons']) for layer in layers]

# Explicitly build a Network so objects stored in container structures are included.
net_objects = []
for layer in layers:
    net_objects.extend([
        layer['exc_neurons'],
        layer['inh_neurons'],
        layer['syn_ee'],
        layer['syn_ii'],
        layer['syn_ei'],
        layer['syn_ie'],
    ])
for inter_syn in interlayer_synapses:
    net_objects.extend([
        inter_syn['syn_ee'],
        inter_syn['syn_ei'],
        inter_syn['syn_ie'],
        inter_syn['syn_ii'],
    ])
net_objects.extend(spike_mon_exc)
net_objects.extend(spike_mon_inh)

Network(net_objects).run(duration)

S_hat_list = []
for layer_i in range(N_layers):
    print(f"  Layer {layer_i} spikes (E): {spike_mon_exc[layer_i].num_spikes}, (I): {spike_mon_inh[layer_i].num_spikes}")

    spike_i_exc = np.asarray(spike_mon_exc[layer_i].i[:], dtype=int)
    spike_t_exc_ms = np.asarray(spike_mon_exc[layer_i].t[:] / ms, dtype=float)
    S, S_shuff_mean, S_minus_Sshuff = compute_S_metrics(
        spike_i_exc,
        spike_t_exc_ms,
        layers[layer_i]['cluster_ids'],
        group_size=N_exc_c,
        bin_size_ms=100.0,
        n_shuffles=10
    )
    print(f"Layer {layer_i} (E) S metrics:")
    print(f"S: {S:.6f}")
    print(f"<S_shuff>: {S_shuff_mean:.6f}")
    print(f"S - <S_shuff>: {S_minus_Sshuff:.6f}")
    S_hat_list.append(S_minus_Sshuff)

with open('output/S_hat_values.txt', 'w') as f:
    f.write("Layer,S_hat\n")
    for layer_i in range(N_layers):
        f.write(f"{layer_i},{S_hat_list[layer_i]:.6f}\n")


def plot_multilayer_3d_structure(layers, interlayer_synapses, output_path):
    """Plot a 3D view of all layers and sampled E-E edges to highlight column structure."""
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    show_connection_lines = False

    layer_spacing_um = 140.0
    cmap = plt.get_cmap('tab10')

    # Keep edge counts limited so the figure stays legible and fast to render.
    max_incolumn_edges_per_column = 420
    max_outcolumn_edges_per_layer = 500
    intralayer_edge_color = '0.20'

    all_positions_um = np.vstack([layer['positions_um'] for layer in layers])
    x_min = float(np.min(all_positions_um[:, 0]))
    x_max = float(np.max(all_positions_um[:, 0]))
    y_min = float(np.min(all_positions_um[:, 1]))
    y_max = float(np.max(all_positions_um[:, 1]))
    x_pad = 15.0
    y_pad = 15.0
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    x_plane = np.array([[x_min, x_max], [x_min, x_max]])
    y_plane = np.array([[y_min, y_min], [y_max, y_max]])

    for layer_i, layer in enumerate(layers):
        z_val = layer_i * layer_spacing_um
        positions_um = layer['positions_um']
        cluster_ids = np.asarray(layer['cluster_ids'], dtype=int)

        # Add a translucent slab and edge frame so layer separation is obvious.
        slab_color = (0.90, 0.94, 0.98) if (layer_i % 2 == 0) else (0.95, 0.95, 0.95)
        z_plane = np.full((2, 2), z_val)
        ax.plot_surface(
            x_plane,
            y_plane,
            z_plane,
            color=slab_color,
            alpha=0.16,
            shade=False,
            linewidth=0,
        )

        frame_xy = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],
        ])
        ax.plot(
            frame_xy[:, 0],
            frame_xy[:, 1],
            np.full(frame_xy.shape[0], z_val),
            color='black',
            alpha=0.35,
            linewidth=0.9,
        )

        unique_cols = np.unique(cluster_ids)
        for c in unique_cols:
            mask = cluster_ids == c
            color = cmap(int(c) % 10)
            ax.scatter(
                positions_um[mask, 0],
                positions_um[mask, 1],
                np.full(np.sum(mask), z_val),
                s=18,
                color=color,
                alpha=0.07,
                edgecolors='none',
            )
            ax.scatter(
                positions_um[mask, 0],
                positions_um[mask, 1],
                np.full(np.sum(mask), z_val),
                s=6,
                color=color,
                alpha=0.88,
                edgecolors='none',
            )
            if np.sum(mask) > 0:
                cx = float(np.mean(positions_um[mask, 0]))
                cy = float(np.mean(positions_um[mask, 1]))
                ax.scatter(
                    [cx],
                    [cy],
                    [z_val],
                    s=42,
                    color=color,
                    edgecolors='black',
                    linewidths=0.3,
                    alpha=0.95,
                )
                ax.text(cx, cy, z_val + 6.0, f'C{int(c)}', fontsize=8, color=color)

        if show_connection_lines:
            syn_ee = layer['syn_ee']
            pre_idx = np.asarray(syn_ee.i[:], dtype=int)
            post_idx = np.asarray(syn_ee.j[:], dtype=int)
            same_column = cluster_ids[pre_idx] == cluster_ids[post_idx]

            for c in unique_cols:
                col_mask = same_column & (cluster_ids[pre_idx] == c)
                edge_ids = np.flatnonzero(col_mask)
                if edge_ids.size == 0:
                    continue
                if edge_ids.size > max_incolumn_edges_per_column:
                    edge_ids = np.random.choice(edge_ids, size=max_incolumn_edges_per_column, replace=False)

                for edge_idx in edge_ids:
                    i_pre = pre_idx[edge_idx]
                    i_post = post_idx[edge_idx]
                    ax.plot(
                        [positions_um[i_pre, 0], positions_um[i_post, 0]],
                        [positions_um[i_pre, 1], positions_um[i_post, 1]],
                        [z_val, z_val],
                        color=intralayer_edge_color,
                        alpha=0.16,
                        linewidth=0.55,
                    )

            # Draw a smaller number of out-of-column edges in gray for contrast.
            out_edge_ids = np.flatnonzero(~same_column)
            if out_edge_ids.size > 0:
                if out_edge_ids.size > max_outcolumn_edges_per_layer:
                    out_edge_ids = np.random.choice(out_edge_ids, size=max_outcolumn_edges_per_layer, replace=False)
                for edge_idx in out_edge_ids:
                    i_pre = pre_idx[edge_idx]
                    i_post = post_idx[edge_idx]
                    ax.plot(
                        [positions_um[i_pre, 0], positions_um[i_post, 0]],
                        [positions_um[i_pre, 1], positions_um[i_post, 1]],
                        [z_val, z_val],
                        color=intralayer_edge_color,
                        alpha=0.13,
                        linewidth=0.50,
                    )

        # Label each layer near the center to make depth ordering explicit.
        ax.text(
            x_max + 4.0,
            y_min,
            z_val,
            f'Layer {layer_i}',
            fontsize=10,
            color='black',
        )

    if show_connection_lines:
        # Show a sparse set of strongest inter-layer E-E links (highest spatially decayed weight).
        max_interlayer_edges_per_pair = 120
        for layer_i, inter_syn in enumerate(interlayer_synapses):
            pre_positions_um = np.asarray(layers[layer_i]['positions_um'], dtype=float)
            post_positions_um = np.asarray(layers[layer_i + 1]['positions_um'], dtype=float)
            z_pre = layer_i * layer_spacing_um
            z_post = (layer_i + 1) * layer_spacing_um

            syn_ee_inter = inter_syn['syn_ee']
            pre_idx = np.asarray(syn_ee_inter.i[:], dtype=int)
            post_idx = np.asarray(syn_ee_inter.j[:], dtype=int)
            w_syn = np.asarray(syn_ee_inter.w_syn[:] / Hz, dtype=float)
            if pre_idx.size == 0:
                continue

            top_n = min(max_interlayer_edges_per_pair, pre_idx.size)
            top_ids = np.argpartition(w_syn, -top_n)[-top_n:]
            top_ids = top_ids[np.argsort(w_syn[top_ids])[::-1]]

            # Normalize for alpha scaling among selected links.
            w_sel = w_syn[top_ids]
            w_min = float(np.min(w_sel))
            w_max = float(np.max(w_sel))
            w_span = max(w_max - w_min, 1e-12)

            for edge_idx in top_ids:
                i_pre = pre_idx[edge_idx]
                i_post = post_idx[edge_idx]
                strength = (w_syn[edge_idx] - w_min) / w_span
                ax.plot(
                    [pre_positions_um[i_pre, 0], post_positions_um[i_post, 0]],
                    [pre_positions_um[i_pre, 1], post_positions_um[i_post, 1]],
                    [z_pre, z_post],
                    color='black',
                    alpha=0.08 + 0.22 * strength,
                    linewidth=0.35 + 0.65 * strength,
                )

    ax.set_xlim(x_min, x_max + 25.0)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(-0.5 * layer_spacing_um, (len(layers) - 1 + 0.5) * layer_spacing_um)
    ax.set_zticks([])
    ax.set_box_aspect((x_max - x_min, y_max - y_min, 1.05 * (len(layers) - 1) * layer_spacing_um))
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_title('5-layer 3D structure')
    ax.view_init(elev=22, azim=-57)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig


def plot_uniform_layer_proxy_columns(layers, output_path, uniform_start_idx):
    """Plot one uniform layer in 2D, color-coded by proxy column ID."""
    uniform_layer_indices = [idx for idx in range(len(layers)) if idx >= uniform_start_idx]
    if not uniform_layer_indices:
        print("No uniform layers found; skipping proxy-column spatial figure.")
        return None

    layer_i = uniform_layer_indices[0]
    layer = layers[layer_i]
    positions_um = np.asarray(layer['positions_um'], dtype=float)
    cluster_ids = np.asarray(layer['cluster_ids'], dtype=int)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    cmap = plt.get_cmap('tab10')

    unique_cols = np.unique(cluster_ids)
    for c in unique_cols:
        mask = cluster_ids == c
        ax.scatter(
            positions_um[mask, 0],
            positions_um[mask, 1],
            s=13,
            alpha=0.85,
            color=cmap(int(c) % 10),
            edgecolors='none',
            label=f'C{int(c)}',
        )

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_title(f'Uniform layer {layer_i}: proxy-column assignment')
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right', ncol=1, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig

# Plot all layers: one row per layer, spatial in col 1 and raster in col 2.
fig, axes = plt.subplots(N_layers, 2, figsize=(16, 5 * N_layers), squeeze=False)
duration_ms = float(duration / ms)

for layer_i in range(N_layers):
    layer = layers[layer_i]
    ax_spatial = axes[layer_i, 0]
    ax_raster = axes[layer_i, 1]

    layer_positions_um = layer['positions_um']
    exc_cluster_ids = np.asarray(layer['cluster_ids'], dtype=int)
    n_exc_layer = int(exc_cluster_ids.size)

    inh_neurons = layer['inh_neurons']
    inh_x = np.array(inh_neurons.x / um)
    inh_y = np.array(inh_neurons.y / um)
    ax_spatial.scatter(inh_x, inh_y, s=8, c='tab:blue', alpha=0.2)
    ax_spatial.scatter(layer_positions_um[:, 0], layer_positions_um[:, 1], s=8, c='tab:red', alpha=0.85)

    layer_centroids = layer['centroids']
    if layer_centroids is not None:
        # Draw one 2-sigma circle around each cluster centroid and label cluster IDs.
        for cluster_idx, centroid in enumerate(layer_centroids):
            cx_um = float(centroid[0] / um)
            cy_um = float(centroid[1] / um)
            circle = Circle((cx_um, cy_um), radius=float(2 * sigma_c / um), fill=False, linestyle='-', linewidth=1.2, edgecolor='tab:green', alpha=0.9)
            ax_spatial.add_patch(circle)
            ax_spatial.text(cx_um, cy_um, f'C{cluster_idx}', color='tab:green', fontsize=9, ha='center', va='center')

    ax_spatial.set_aspect('equal', adjustable='box')
    ax_spatial.set_xlabel('x (um)')
    ax_spatial.set_ylabel('y (um)')
    ax_spatial.set_title(f'Layer {layer_i}: Spatial connectivity graph')
    ax_spatial.autoscale()

    layer_spike_mon_exc = spike_mon_exc[layer_i]
    layer_spike_mon_inh = spike_mon_inh[layer_i]

    # Plot-only remap: group excitatory raster rows by (proxy) column id.
    exc_sort_order = np.argsort(exc_cluster_ids, kind='stable')
    exc_row_map = np.empty(n_exc_layer, dtype=int)
    exc_row_map[exc_sort_order] = np.arange(n_exc_layer)
    exc_spike_rows = exc_row_map[np.asarray(layer_spike_mon_exc.i[:], dtype=int)]

    ax_raster.scatter(
        layer_spike_mon_exc.t / ms,
        exc_spike_rows,
        s=2,
        c='tab:red',
        alpha=0.7,
        label='Excitatory'
    )
    ax_raster.scatter(
        layer_spike_mon_inh.t / ms,
        n_exc_layer + layer_spike_mon_inh.i,
        s=2,
        c='tab:blue',
        alpha=0.7,
        label='Inhibitory'
    )
    ax_raster.set_xlabel('Time (ms)')
    ax_raster.set_ylabel('Neuron index')
    ax_raster.set_title(f'Layer {layer_i}: Spike raster')
    ax_raster.legend(loc='upper right', markerscale=3)

    # Label excitatory index ranges by (proxy) column ID using actual group sizes.
    ax_raster.set_xlim(0.0, duration_ms)
    sorted_ids = exc_cluster_ids[exc_sort_order]
    unique_ids, counts = np.unique(sorted_ids, return_counts=True)
    y_cursor = 0
    for block_idx, (cluster_idx, count) in enumerate(zip(unique_ids, counts)):
        y_start = y_cursor
        y_end = y_cursor + int(count)
        y_center = 0.5 * (y_start + y_end - 1)
        if block_idx % 2 == 0:
            ax_raster.axhspan(y_start, y_end, color='gray', alpha=0.04)
        # ax_raster.axhline(y_start, color='gray', linewidth=0.4, alpha=0.5)
        ax_raster.text(
            1.02,
            y_center,
            f'C{int(cluster_idx)}',
            transform=ax_raster.get_yaxis_transform(),
            fontsize=8,
            color='black',
            ha='left',
            va='center',
            clip_on=False,
        )
        y_cursor = y_end

    ax_raster.axhline(n_exc_layer, color='gray', linewidth=0.6, alpha=0.8)
    if layer['centroids'] is None:
        ax_raster.text(1.02, 0.5 * n_exc_layer, 'readout', transform=ax_raster.get_yaxis_transform(), fontsize=8, color='tab:red', ha='left', va='center', clip_on=False)
    ax_raster.text(1.02, n_exc_layer + 0.5 * N_inh, 'inh', transform=ax_raster.get_yaxis_transform(), fontsize=8, color='tab:blue', ha='left', va='center', clip_on=False)
    ax_raster.set_ylim(-1, n_exc_layer + N_inh + 1)

fig.text(
	0.01,
	0.01,
	f"Pentacle radius: {float(R / um):.1f} um | "
	f"Cluster radius (stdev): {float(sigma_c / um):.1f} um | "
    f"R_ee: {R_ee:.2f} | "
    f"Layer 0 p_ee_in/out: {layers[0]['p_ee_in']:.3f}/{layers[0]['p_ee_out']:.3f} | "
	f"Random seed: {RANDOM_SEED}",
	fontsize=9,
	ha='left',
	va='bottom'
)
plt.tight_layout(rect=(0.0, 0.06, 0.96, 1.0))
plt.savefig('output/spatial_and_raster_all_layers.png', dpi=300)

Path('output').mkdir(parents=True, exist_ok=True)
plot_multilayer_3d_structure(
    layers,
    interlayer_synapses,
    output_path='output/spatial_structure_3d_columns.png',
)

plot_uniform_layer_proxy_columns(
    layers,
    output_path='output/uniform_layer_proxy_columns.png',
    uniform_start_idx=uniform_layer_start,
)

print('Saved: output/spatial_structure_3d_columns.png')
print('Saved: output/uniform_layer_proxy_columns.png')
plt.show()

