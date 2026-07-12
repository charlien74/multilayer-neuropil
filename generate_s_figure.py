from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_s_hat_rows(csv_path):
	"""Load rows as (layer, s_hat, n_structured_layers|None)."""
	rows = []
	with open(csv_path, "r", encoding="utf-8") as f:
		for line_idx, raw in enumerate(f):
			line = raw.strip()
			if not line:
				continue
			if line_idx == 0 and line.lower().startswith("layer"):
				continue

			parts = [p.strip() for p in line.split(",")]
			if len(parts) < 2:
				continue

			layer = int(parts[0])
			s_hat = float(parts[1])
			n_struct = int(parts[2]) if len(parts) >= 3 else None
			rows.append((layer, s_hat, n_struct))

	if not rows:
		raise ValueError(f"No data rows found in {csv_path}")

	return rows


def resolve_input_path(preferred_path):
	"""Use preferred path if present, else fallback to common legacy filename."""
	if preferred_path.exists():
		return preferred_path
	fallback = Path("output/S_hat_values_uniform.txt")
	if fallback.exists():
		print(f"Input not found at {preferred_path}, using {fallback} instead.")
		return fallback
	raise FileNotFoundError(f"Could not find {preferred_path} or {fallback}")


def plot_grouped_curves(groups, save_path):
	"""Plot all curves on one axes with colors based only on structured vs uniform."""
	fig, ax = plt.subplots(figsize=(9.0, 6.0))

	keys = sorted(groups.keys(), key=lambda k: (-1 if k is None else k))
	structured_color = "tab:blue"
	uniform_color = "tab:red"

	all_layers = []
	for key in keys:
		series = groups[key]
		layers = np.asarray([x[0] for x in series], dtype=int)
		s_hats = np.asarray([x[1] for x in series], dtype=float)
		order = np.argsort(layers)
		layers = layers[order]
		s_hats = s_hats[order]

		if key is None:
			# If split metadata is unavailable, treat the whole curve as structured.
			ax.plot(layers, s_hats, "o-", linewidth=1.8, markersize=4.8, color=structured_color, alpha=0.6)
			all_layers.extend(layers.tolist())
			continue

		structured_mask = layers < key
		uniform_mask = ~structured_mask

		if np.any(structured_mask):
			ax.plot(
				layers[structured_mask],
				s_hats[structured_mask],
				"o-",
				linewidth=1.8,
				markersize=4.8,
				color=structured_color,
				alpha=0.6,
			)

		if np.any(uniform_mask):
			ax.plot(
				layers[uniform_mask],
				s_hats[uniform_mask],
				"o-",
				linewidth=1.8,
				markersize=4.8,
				color=uniform_color,
				alpha=0.6,
			)

		if np.any(structured_mask) and np.any(uniform_mask):
			left_idx = np.where(structured_mask)[0][-1]
			right_idx = np.where(uniform_mask)[0][0]
			x_left = float(layers[left_idx])
			x_right = float(layers[right_idx])
			y_left = float(s_hats[left_idx])
			y_right = float(s_hats[right_idx])
			x_mid = 0.5 * (x_left + x_right)
			y_mid = 0.5 * (y_left + y_right)

			# Switch color midway between the boundary points.
			ax.plot(
				[x_left, x_mid],
				[y_left, y_mid],
				"-",
				linewidth=1.8,
				color=structured_color,
				alpha=0.6,
			)
			ax.plot(
				[x_mid, x_right],
				[y_mid, y_right],
				"-",
				linewidth=1.8,
				color=uniform_color,
				alpha=0.6,
			)
		all_layers.extend(layers.tolist())

	ax.set_xlabel("Layer index")
	ax.set_ylabel(r"$\hat{S}$")
	ax.set_title(r"$\hat{S}$ vs Layer Index ($R_{ee} = 1.0$)")
	ax.grid(alpha=0.25)
	if all_layers:
		ax.set_xticks(np.unique(np.asarray(all_layers, dtype=int)))

	legend_handles = [
		Line2D([0], [0], color=structured_color, lw=2, label="Structured Layers"),
		Line2D([0], [0], color=uniform_color, lw=2, label="Unstructured Layers"),
	]
	ax.legend(handles=legend_handles, loc="best")

	fig.tight_layout()
	fig.savefig(save_path, dpi=300)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description="Generate S_hat-vs-layer figure with all uniform-sweep curves on shared axes.")
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("output/S_hat_uniform.csv"),
		help="Path to CSV containing Layer,S_hat[,n_structured_layers].",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("output/S_hat_uniform_multiline.png"),
		help="Output image path.",
	)
	args = parser.parse_args()

	input_path = resolve_input_path(args.input)
	rows = load_s_hat_rows(input_path)

	# If multiple n_structured_layers values are present, plot each in its own file.
	groups = {}
	for layer, s_hat, n_struct in rows:
		groups.setdefault(n_struct, []).append((layer, s_hat))

	args.output.parent.mkdir(parents=True, exist_ok=True)

	plot_grouped_curves(groups, args.output)
	print(f"Saved: {args.output}")


if __name__ == "__main__":
	main()
