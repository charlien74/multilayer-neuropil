from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
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

p_avg=0.01

R_ee = 1.2
interlayer_decay_l = 30 * um
inhibitory_sigma = R / 2
num_exc_per_layer = N_exc_c * 5  # Number of excitatory neurons per layer
uniform_radius = (R + 3 * sigma_c) / um  # Ensure neurons are within a reasonable distance from the center

layers = []
for layer_i in range(N_layers):
    centroids, neuron_locations, positions_um, cluster_ids = generate_pentacle_layout(
        assembly_radius=sigma_c, pentacle_radius=R, n_clusters=5, neurons_per_cluster=N_exc_c) 
    inh_neuron_locations, inh_positions_um, inh_cluster_ids = generate_inhibitory_locations(assembly_radius=inhibitory_sigma, n_inhibitory=N_inh)

    n_clusters = np.bincount(cluster_ids)
    largest_community_size = int(n_clusters.max()) if n_clusters.size > 0 else num_exc_per_layer
    p_ee_in, p_ee_out = get_p_connection_in_out(
        p_ee_avg=p_avg,
        R_ee=R_ee,
        N_excitatory=num_exc_per_layer,
        cluster_size=largest_community_size,
    )
    ## Remove assemblies from top layer 
    if layer_i == N_layers - 1:
        neuron_locations, positions_um, cluster_ids = generate_uniform_layout(radius=uniform_radius, n_neurons=1600)
        inh_neuron_locations, inh_positions_um, _ = generate_uniform_layout(radius=uniform_radius, n_neurons=N_inh)
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
        "centroids": centroids if layer_i != N_layers - 1 else None,
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
    if layer_i != N_layers - 1:
        # Pentacle layers with clusters
        for cluster_idx in range(5):
            y_start = cluster_idx * N_exc_c
            y_end = (cluster_idx + 1) * N_exc_c
            y_center = 0.5 * (y_start + y_end - 1)
            if cluster_idx % 2 == 0:
                ax_raster.axhspan(y_start, y_end, color='gray', alpha=0.04)
            ax_raster.axhline(y_start, color='gray', linewidth=0.4, alpha=0.5)
            ax_raster.text(1.02, y_center, f'C{cluster_idx}', transform=ax_raster.get_yaxis_transform(), fontsize=8, color='black', ha='left', va='center', clip_on=False)
        ax_raster.axhline(5 * N_exc_c, color='gray', linewidth=0.6, alpha=0.8)
    else:
        # Readout layer (uniform)
        ax_raster.axhline(num_exc_per_layer, color='gray', linewidth=0.6, alpha=0.8)
        ax_raster.text(1.02, 0.5 * num_exc_per_layer, 'readout', transform=ax_raster.get_yaxis_transform(), fontsize=8, color='tab:red', ha='left', va='center', clip_on=False)
    ax_raster.text(1.02, num_exc_per_layer + 0.5 * N_inh, 'inh', transform=ax_raster.get_yaxis_transform(), fontsize=8, color='tab:blue', ha='left', va='center', clip_on=False)
    ax_raster.set_ylim(-1, num_exc_per_layer + N_inh + 1)

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
plt.show()

