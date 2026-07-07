from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from model_util import *

start_scope()
seed(RANDOM_SEED)
duration = 20 * second

n_clusters = 10
n_exc_per_cluster = 320
n_inh = 400
cluster_sigma = 10 * um
cluster_distance = 40 * um

p_max_exc = 0.28
p_max_inh = 0.2
sigma_connection = 40 * um

neuron_locations = []
cluster_indices = []
cluster_centers = []
for cluster_idx in range(n_clusters):
    cluster_center = cluster_distance * (cluster_idx - floor(n_clusters / 2))
    cluster_centers.append(cluster_center)
    for _ in range(n_exc_per_cluster):
        x = cluster_center + np.random.normal(0.0, float(cluster_sigma / um)) * um
        neuron_locations.append(x)
        cluster_indices.append(cluster_idx)

exc_neurons = NeuronGroup(
    n_clusters * n_exc_per_cluster,
    eqs_exc,
    method="euler",
    threshold="v > v_th",
    reset="v = v_reset",
    refractory=refractory,
    name="exc_neurons")

inh_neurons = NeuronGroup(
    n_inh,
    eqs_inh,
    method="euler",
    threshold="v > v_th",
    reset="v = v_reset",
    refractory=refractory,
    name="inh_neurons")

exc_neurons.tau_m = tau_m_e
exc_neurons.mu = "1.1 + 0.1*rand()"
exc_neurons.w_ee = 0.0156 * kHz
exc_neurons.w_ei = -0.0297 * kHz
exc_neurons.v = "rand()"
exc_neurons.g_e = 0
exc_neurons.g_i = 0

inh_neurons.tau_m = tau_m_i
inh_neurons.mu = "1 + 0.05*rand()"
inh_neurons.w_ie = 0.0074 * kHz
inh_neurons.w_ii = -0.0297 * kHz
inh_neurons.v = "rand()"
inh_neurons.g_e = 0
inh_neurons.g_i = 0

inh_min_um = float((cluster_centers[0] - cluster_sigma * 2) / um)
inh_max_um = float((cluster_centers[-1] + cluster_sigma * 2) / um)
inh_neurons.x = np.random.uniform(inh_min_um, inh_max_um, n_inh) * um
# One-dimensional layout: all neurons lie on y=0.
inh_neurons.y = 0 * um

exc_neurons.x = np.array([float(x / um) for x in neuron_locations], dtype=float) * um
exc_neurons.y = 0 * um
exc_neurons.column_id = np.array(cluster_indices)

syn_ee = Synapses(exc_neurons, exc_neurons, on_pre="g_e_post += 1")
syn_ee.connect(condition="i != j", p='clip(p_max_exc * exp(-abs(x_pre - x_post) / sigma_connection), 0, 1)')
syn_ii = Synapses(inh_neurons, inh_neurons, on_pre="g_i_post += 1")
syn_ii.connect(condition="i != j", p=0.5)

syn_ei = Synapses(inh_neurons, exc_neurons, on_pre="g_i_post += 1")
syn_ei.connect(p=0.5)

syn_ie = Synapses(exc_neurons, inh_neurons, on_pre="g_e_post += 1")
syn_ie.connect(p=0.5)

spike_mon_exc = SpikeMonitor(exc_neurons)
spike_mon_inh = SpikeMonitor(inh_neurons)

net = Network(
    exc_neurons,
    inh_neurons,
    syn_ee,
    syn_ii,
    syn_ei,
    syn_ie,
    spike_mon_exc,
    spike_mon_inh,
)
net.run(duration)

print(f"Excitatory spikes: {spike_mon_exc.num_spikes}")
print(f"Inhibitory spikes: {spike_mon_inh.num_spikes}")

duration_ms = float(duration / ms)
x_exc_um = np.asarray(exc_neurons.x[:] / um, dtype=float)
cluster_ids = np.asarray(exc_neurons.column_id[:], dtype=int)
centers_um = np.asarray([float(c / um) for c in cluster_centers], dtype=float)
sigma_um = float(cluster_sigma / um)

x_margin = 4.0 * sigma_um
x_min = centers_um.min() - x_margin
x_max = centers_um.max() + x_margin
x_grid = np.linspace(x_min, x_max, 1000)

gaussian_curves = []
for center_um in centers_um:
    density = (1.0 / (sigma_um * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x_grid - center_um) / sigma_um) ** 2)
    gaussian_curves.append(density)
max_density = max(np.max(curve) for curve in gaussian_curves)
dot_y = -0.06 * max_density
dot_jitter = 0.01 * max_density
rng = np.random.default_rng(RANDOM_SEED)

fig, (ax_gauss, ax_raster) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
for cluster_idx in range(n_clusters):
    color = cluster_colors[cluster_idx]
    curve = gaussian_curves[cluster_idx]
    cluster_x = x_exc_um[cluster_ids == cluster_idx]
    cluster_y = dot_y - rng.uniform(0.0, dot_jitter, size=cluster_x.shape)
    ax_gauss.plot(x_grid, curve, color=color, linewidth=2, label=f"Cluster {cluster_idx} Gaussian")
    ax_gauss.scatter(cluster_x, cluster_y, s=8, color="red", alpha=0.45)

ax_gauss.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
ax_gauss.set_xlabel("x position (um)")
ax_gauss.set_ylabel("Probability density")
ax_gauss.set_title("1D cluster Gaussians with sampled neuron locations (red dots)")
ax_gauss.set_xlim(x_min, x_max)
ax_gauss.set_ylim(dot_y * 1.6, max_density * 1.08)
ax_gauss.legend(loc="upper right", ncol=2, fontsize=9)

ax_raster.scatter(spike_mon_exc.t / ms, spike_mon_exc.i, s=2, c="tab:red", alpha=0.7, label="Excitatory")
ax_raster.scatter(
    spike_mon_inh.t / ms,
    n_clusters * n_exc_per_cluster + spike_mon_inh.i,
    s=2,
    c="tab:blue",
    alpha=0.7,
    label="Inhibitory",
)
ax_raster.set_xlim(0.0, duration_ms)
ax_raster.set_ylim(-1, n_clusters * n_exc_per_cluster + n_inh + 1)
ax_raster.set_xlabel("Time (ms)")
ax_raster.set_ylabel("Neuron index")
ax_raster.set_title("Full-duration spike raster")
ax_raster.legend(loc="upper right", markerscale=3)

for cluster_idx in range(n_clusters):
    y_start = cluster_idx * n_exc_per_cluster
    y_end = (cluster_idx + 1) * n_exc_per_cluster
    if cluster_idx % 2 == 0:
        ax_raster.axhspan(y_start, y_end, color="gray", alpha=0.05)
    ax_raster.axhline(y_start, color="gray", linewidth=0.4, alpha=0.6)

ax_raster.axhline(n_clusters * n_exc_per_cluster, color="gray", linewidth=0.8, alpha=0.8)

# Animate COM and spread of spike locations over a sliding 100 ms bucket.
bucket_ms = 100.0
exc_spike_t_ms = np.asarray(spike_mon_exc.t / ms, dtype=float)
exc_spike_x_um = x_exc_um[np.asarray(spike_mon_exc.i[:], dtype=int)]
inh_x_um = np.asarray(inh_neurons.x[:] / um, dtype=float)
inh_spike_t_ms = np.asarray(spike_mon_inh.t / ms, dtype=float)
inh_spike_x_um = inh_x_um[np.asarray(spike_mon_inh.i[:], dtype=int)]

com_spike_t_ms = exc_spike_t_ms
com_spike_x_um = exc_spike_x_um

bucket_starts = np.arange(0.0, duration_ms, bucket_ms)
bucket_ends = np.minimum(bucket_starts + bucket_ms, duration_ms)
bucket_com_um = np.full(bucket_starts.shape, np.nan)
bucket_std_um = np.full(bucket_starts.shape, np.nan)
bucket_counts = np.zeros(bucket_starts.shape, dtype=int)

for bucket_idx, (t_start, t_end) in enumerate(zip(bucket_starts, bucket_ends)):
    in_bucket = (com_spike_t_ms >= t_start) & (com_spike_t_ms < t_end)
    x_bucket = com_spike_x_um[in_bucket]
    bucket_counts[bucket_idx] = x_bucket.size
    if x_bucket.size > 0:
        bucket_com_um[bucket_idx] = float(np.mean(x_bucket))
        bucket_std_um[bucket_idx] = float(np.std(x_bucket))

fig_anim, (ax_anim, ax_anim_raster) = plt.subplots(
    2,
    1,
    figsize=(14, 9),
    constrained_layout=True,
    gridspec_kw={"height_ratios": [1.0, 1.35]},
)
for cluster_idx in range(n_clusters):
    color = cluster_colors[cluster_idx]
    curve = gaussian_curves[cluster_idx]
    cluster_x = x_exc_um[cluster_ids == cluster_idx]
    cluster_y = dot_y - rng.uniform(0.0, dot_jitter, size=cluster_x.shape)
    ax_anim.plot(x_grid, curve, color=color, linewidth=2)
    ax_anim.scatter(cluster_x, cluster_y, s=7, color="red", alpha=0.35)

ax_anim.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
ax_anim.set_xlabel("x position (um)")
ax_anim.set_ylabel("Probability density")
ax_anim.set_title("Spike-position COM and spread over last 100 ms")
ax_anim.set_xlim(x_min, x_max)
ax_anim.set_ylim(dot_y * 2.0, max_density * 1.08)

y_com = max_density * 0.90
y_std = max_density * 0.82
com_marker, = ax_anim.plot([], [], marker="o", linestyle="None", color="black", markersize=8, label="COM")
std_span, = ax_anim.plot([], [], color="black", linewidth=3, alpha=0.9, label="COM ± 1σ")
window_text = ax_anim.text(0.02, 0.97, "", transform=ax_anim.transAxes, ha="left", va="top")
count_text = ax_anim.text(0.02, 0.90, "", transform=ax_anim.transAxes, ha="left", va="top")
ax_anim.legend(loc="upper right")

ax_anim_raster.scatter(
    exc_spike_t_ms,
    np.asarray(spike_mon_exc.i[:], dtype=int),
    s=1.2,
    c="tab:red",
    alpha=0.6,
    label="Excitatory",
)
ax_anim_raster.scatter(
    inh_spike_t_ms,
    n_clusters * n_exc_per_cluster + np.asarray(spike_mon_inh.i[:], dtype=int),
    s=1.2,
    c="tab:blue",
    alpha=0.6,
    label="Inhibitory",
)
ax_anim_raster.set_xlim(0.0, duration_ms)
ax_anim_raster.set_ylim(-1, n_clusters * n_exc_per_cluster + n_inh + 1)
ax_anim_raster.set_xlabel("Time (ms)")
ax_anim_raster.set_ylabel("Neuron index")
ax_anim_raster.set_title("Raster with synchronized time cursor")
ax_anim_raster.legend(loc="upper right", markerscale=3)

for cluster_idx in range(n_clusters):
    y_start = cluster_idx * n_exc_per_cluster
    y_end = (cluster_idx + 1) * n_exc_per_cluster
    if cluster_idx % 2 == 0:
        ax_anim_raster.axhspan(y_start, y_end, color="gray", alpha=0.04)
    ax_anim_raster.axhline(y_start, color="gray", linewidth=0.4, alpha=0.5)
ax_anim_raster.axhline(n_clusters * n_exc_per_cluster, color="gray", linewidth=0.8, alpha=0.75)

time_cursor = ax_anim_raster.axvline(0.0, color="black", linewidth=1.6, alpha=0.95)

def init_anim():
    com_marker.set_data([], [])
    std_span.set_data([], [])
    time_cursor.set_xdata([0.0, 0.0])
    window_text.set_text("")
    count_text.set_text("")
    return com_marker, std_span, time_cursor, window_text, count_text

def update_anim(frame_idx):
    com_um = bucket_com_um[frame_idx]
    std_um = bucket_std_um[frame_idx]
    t_start = bucket_starts[frame_idx]
    t_end = bucket_ends[frame_idx]
    count = bucket_counts[frame_idx]

    if np.isfinite(com_um):
        com_marker.set_data([com_um], [y_com])
        std_span.set_data([com_um - std_um, com_um + std_um], [y_std, y_std])
    else:
        com_marker.set_data([], [])
        std_span.set_data([], [])

    time_cursor.set_xdata([t_end, t_end])

    window_text.set_text(f"Window: [{t_start:.0f}, {t_end:.0f}) ms")
    count_text.set_text(f"Spikes in window: {count}")
    return com_marker, std_span, time_cursor, window_text, count_text

com_anim = animation.FuncAnimation(
    fig_anim,
    update_anim,
    init_func=init_anim,
    frames=len(bucket_starts),
    interval=180,
    blit=True,
    repeat=True,
)

Path("output").mkdir(parents=True, exist_ok=True)
fig.savefig("output/one_d_gaussian_and_raster.png", dpi=300)
animation_path = Path("output") / "one_d_gaussian_com_animation.gif"
try:
    com_anim.save(animation_path, writer=animation.PillowWriter(fps=6))
    print(f"Saved animation: {animation_path}")
except Exception as exc:
    print(f"Could not save animation ({exc}).")
plt.show()
