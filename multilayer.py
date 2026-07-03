from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from model_util import *

start_scope()
seed(RANDOM_SEED)

N_layers = 5

TARGET_AVG_P_OVERALL = 0.025
max_p_exc_interlayer = 0.25
max_p_inh_interlayer = 0.25
interlayer_decay_l = 20 * um
num_exc_per_layer = N_exc_c * 5  # Number of excitatory neurons per layer
uniform_radius = (R + 3 * sigma_c) / um  # Ensure neurons are within a reasonable distance from the center

layers = []
for layer_i in range(N_layers):
    p_max_exc, centroids, neuron_locations, positions_um, cluster_ids = get_spatial_assembly_layout_target_p_avg(
        assembly_radius=sigma_c,
        pentacle_radius=R,
        sigma_connection=sigma_connection,
        target_overall_avg=TARGET_AVG_P_OVERALL,
        n_clusters=5,
        neurons_per_cluster=N_exc_c,
    )
    ## Remove assemblies from top layer 
    if layer_i == N_layers - 1:
        neuron_locations, positions_um, cluster_ids = generate_uniform_layout(radius=uniform_radius, n_neurons=1600)
        num_exc_per_layer = len(neuron_locations)  # Update the number of excitatory neurons for the top layer
    layer_exc_neurons = NeuronGroup(
        num_exc_per_layer,
        eqs_exc,
        threshold='v > v_th',
        reset='v = v_reset',
        refractory=refractory,
        method='euler'
    )
    layer_inh_neurons = inh_neurons = NeuronGroup(
        N_inh,
        eqs_inh,
        threshold='v > v_th',
        reset='v = v_reset',
        refractory=refractory,
        method='euler'
    )

    layer_exc_neurons.tau_m = tau_m_e
    layer_exc_neurons.mu = "1.1 + 0.1*rand()"
    layer_exc_neurons.w_ee = 0.0156 * kHz
    layer_exc_neurons.w_ei = -0.0297 * kHz
    layer_exc_neurons.v = "rand()"
    layer_exc_neurons.g_e = 0
    layer_exc_neurons.g_i = 0

    layer_inh_neurons.tau_m = tau_m_i
    layer_inh_neurons.mu = "1 + 0.05*rand()"
    layer_inh_neurons.w_ie = 0.0074 * kHz
    layer_inh_neurons.w_ii = -0.0297 * kHz
    layer_inh_neurons.v = "rand()"
    layer_inh_neurons.g_e = 0
    layer_inh_neurons.g_i = 0

    x_coords_um = positions_um[:, 0]
    y_coords_um = positions_um[:, 1]
    layer_exc_neurons.x = x_coords_um * um
    layer_exc_neurons.y = y_coords_um * um
    layer_exc_neurons.column_id = cluster_ids

    # Distance-decaying random connectivity:
    # p(d) = p_max_exc * exp(-distance / sigma_connection)
    syn_ee = Synapses(layer_exc_neurons, layer_exc_neurons, on_pre="g_e_post += 1")
    syn_ee.connect(
        condition='i != j',
        p='clip(p_max_exc * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / sigma_connection), 0, 1)'
    )

    syn_ii = Synapses(layer_inh_neurons, layer_inh_neurons, on_pre="g_i_post += 1")
    syn_ii.connect(condition="i != j", p=0.5)

    syn_ei = Synapses(layer_inh_neurons, layer_exc_neurons, on_pre="g_i_post += 1")
    syn_ei.connect(p=0.5)

    syn_ie = Synapses(layer_exc_neurons, layer_inh_neurons, on_pre="g_e_post += 1")
    syn_ie.connect(p=0.5)

    layers.append({
        "exc_neurons": layer_exc_neurons,
        "inh_neurons": layer_inh_neurons,
        "syn_ee": syn_ee,
        "syn_ii": syn_ii,
        "syn_ei": syn_ei,
        "syn_ie": syn_ie,
        "positions_um": positions_um,
        "centroids": centroids if layer_i != N_layers - 1 else None,
        "cluster_ids": cluster_ids
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
    syn_ee_inter = Synapses(pre_exc, post_exc, on_pre="g_e_post += 1")
    syn_ee_inter.connect(condition="True", p='max_p_exc_interlayer * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / interlayer_decay_l)')
    syn_ei_inter = Synapses(pre_exc, post_inh, on_pre="g_e_post += 1")
    syn_ei_inter.connect(condition="True", p='max_p_exc_interlayer * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / interlayer_decay_l)')
    syn_ie_inter = Synapses(pre_inh, post_exc, on_pre="g_i_post += 1")
    syn_ie_inter.connect(condition="True", p='max_p_inh_interlayer * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / interlayer_decay_l)')
    syn_ii_inter = Synapses(pre_inh, post_inh, on_pre="g_i_post += 1")
    syn_ii_inter.connect(condition="True", p='max_p_inh_interlayer * exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / interlayer_decay_l)')
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

for layer_i in range(N_layers):
    print(f"  Layer {layer_i} spikes (E): {spike_mon_exc[layer_i].num_spikes}, (I): {spike_mon_inh[layer_i].num_spikes}")

spike_i_exc = np.asarray(spike_mon_exc[N_layers - 1].i[:], dtype=int)
spike_t_exc_ms = np.asarray(spike_mon_exc[N_layers - 1].t[:] / ms, dtype=float)
S, S_shuff_mean, S_minus_Sshuff = compute_S_metrics(
	spike_i_exc,
	spike_t_exc_ms,
	layers[N_layers - 1]['cluster_ids'],
	group_size=N_exc_c,
	bin_size_ms=100.0,
	n_shuffles=10
)
print(f"S: {S:.6f}")
print(f"<S_shuff>: {S_shuff_mean:.6f}")
print(f"S - <S_shuff>: {S_minus_Sshuff:.6f}")

# Plot all layers: one row per layer, spatial in col 1 and raster in col 2.
fig, axes = plt.subplots(N_layers, 2, figsize=(16, 5 * N_layers), squeeze=False)
duration_ms = float(duration / ms)

for layer_i in range(N_layers):
    layer = layers[layer_i]
    ax_spatial = axes[layer_i, 0]
    ax_raster = axes[layer_i, 1]

    layer_positions_um = layer['positions_um']
    src_idx = np.asarray(layer['syn_ee'].i[:], dtype=int)
    dst_idx = np.asarray(layer['syn_ee'].j[:], dtype=int)
    if src_idx.size > 0:
        segments = np.stack((layer_positions_um[src_idx], layer_positions_um[dst_idx]), axis=1)
    else:
        segments = np.empty((0, 2, 2), dtype=float)

    line_collection = LineCollection(segments, colors='k', linewidths=0.2, alpha=0.12)
    ax_spatial.add_collection(line_collection)
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
    ax_raster.scatter(
        layer_spike_mon_exc.t / ms,
        layer_spike_mon_exc.i,
        s=2,
        c='tab:red',
        alpha=0.7,
        label='Excitatory'
    )
    ax_raster.scatter(
        layer_spike_mon_inh.t / ms,
        num_exc_per_layer + layer_spike_mon_inh.i,
        s=2,
        c='tab:blue',
        alpha=0.7,
        label='Inhibitory'
    )
    ax_raster.set_xlabel('Time (ms)')
    ax_raster.set_ylabel('Neuron index')
    ax_raster.set_title(f'Layer {layer_i}: Spike raster')
    ax_raster.legend(loc='upper right', markerscale=3)

    # Label excitatory index ranges by cluster ID.
    ax_raster.set_xlim(0.0, duration_ms)
    for cluster_idx in range(5):
        y_start = cluster_idx * N_exc_c
        y_end = (cluster_idx + 1) * N_exc_c
        y_center = 0.5 * (y_start + y_end - 1)
        if cluster_idx % 2 == 0:
            ax_raster.axhspan(y_start, y_end, color='gray', alpha=0.04)
        ax_raster.axhline(y_start, color='gray', linewidth=0.4, alpha=0.5)
        ax_raster.text(1.02, y_center, f'C{cluster_idx}', transform=ax_raster.get_yaxis_transform(), fontsize=8, color='black', ha='left', va='center', clip_on=False)
    ax_raster.axhline(5 * N_exc_c, color='gray', linewidth=0.6, alpha=0.8)
    ax_raster.text(1.02, num_exc_per_layer + 0.5 * N_inh, 'inh', transform=ax_raster.get_yaxis_transform(), fontsize=8, color='tab:blue', ha='left', va='center', clip_on=False)
    ax_raster.set_ylim(-1, num_exc_per_layer + N_inh + 1)

fig.text(
	0.01,
	0.01,
	f"Pentacle radius: {float(R / um):.1f} um | "
	f"Cluster radius (stdev): {float(sigma_c / um):.1f} um | "
	f"Max excitatory connection probability: {p_max_exc:.3f} | "
	f"Random seed: {RANDOM_SEED}",
	fontsize=9,
	ha='left',
	va='bottom'
)
plt.tight_layout(rect=(0.0, 0.06, 0.96, 1.0))
plt.savefig('output/spatial_and_raster_all_layers.png', dpi=300)
plt.show()

