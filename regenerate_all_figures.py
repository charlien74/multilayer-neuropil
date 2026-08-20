import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd, label):
    print(f"\n[{label}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(
    radius_um,
    layer_weight_decay_lambda,
    summary_signal,
    summary_dt_ms,
    mi_bin_width_ms,
    mi_lag_ms,
    distance_based_mi_h,
    duration_ms,
    r_ee,
    louvain_resolution,
    output_dir,
):
    repo_root = Path(__file__).resolve().parent
    python_exec = sys.executable
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_public_dir = output_dir / 'public'
    output_internal_dir = output_dir / 'internal'
    output_public_dir.mkdir(parents=True, exist_ok=True)
    output_internal_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using Python interpreter: {python_exec}")
    print(f"Repository root: {repo_root}")
    print(f"Output root: {output_dir}")
    print("Regenerating all figures with fresh multilayer internal data...")

    run_step(
        [
            python_exec,
            str(repo_root / 'multilayer.py'),
            '--no-show',
            '--r-ee',
            str(r_ee),
            '--summary-signal',
            summary_signal,
            '--summary-dt-ms',
            str(summary_dt_ms),
            '--mi-bin-width-ms',
            str(mi_bin_width_ms),
            '--duration-ms',
            str(duration_ms),
            '--output-public-dir',
            str(output_public_dir),
            '--output-internal-dir',
            str(output_internal_dir),
        ],
        '1/3 multilayer',
    )
    run_step(
        [
            python_exec,
            str(repo_root / 'neuropil.py'),
            '--radius-um',
            str(radius_um),
            '--layer-weight-decay-lambda',
            str(layer_weight_decay_lambda),
            '--mi-bin-width-ms',
            str(mi_bin_width_ms),
            '--duration-ms',
            str(duration_ms),
            '--output-public-dir',
            str(output_public_dir),
            '--output-internal-dir',
            str(output_internal_dir),
        ],
        '2/3 neuropil',
    )
    run_step(
        [
            python_exec,
            str(repo_root / 'network_analysis.py'),
            '--input-internal-dir',
            str(output_internal_dir),
            '--output-internal-dir',
            str(output_internal_dir),
            '--output-file',
            str(output_public_dir / 'network_compare_results.txt'),
            '--r-ee',
            str(r_ee),
            '--summary-signal',
            summary_signal,
            '--summary-dt-ms',
            str(summary_dt_ms),
            '--layer-weight-decay-lambda',
            str(layer_weight_decay_lambda),
            '--radius-um',
            str(radius_um),
            '--duration-ms',
            str(duration_ms),
            '--mi-lag-ms',
            str(mi_lag_ms),
            '--louvain-resolution',
            str(louvain_resolution),
            '--distance-based-mi-h',
            str(distance_based_mi_h),

        ],
        '3/3 network analysis',
    )

    print("\nDone.")
    print(f"Fresh outputs were generated in {output_public_dir} and {output_internal_dir}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Re-run multilayer and neuropil end-to-end to regenerate figures from fresh internal data.'
    )
    parser.add_argument(
        '--radius-um',
        type=float,
        default=25.0,
        help='Neighborhood radius (um) passed to neuropil.py.',
    )
    parser.add_argument(
        '--layer-weight-decay-lambda',
        type=float,
        default=1.0,
        help='Exponential decay rate for source-layer weighting in neuropil.py.',
    )
    parser.add_argument(
        '--summary-signal',
        choices=['v', 'i_syn', 'g_e'],
        default='v',
        help='Signal exported by multilayer.py for neuropil readout construction.',
    )
    parser.add_argument(
        '--summary-dt-ms',
        type=float,
        default=0.1,
        help='Sampling timestep (ms) for multilayer summary export.',
    )
    parser.add_argument(
        '--mi-bin-width-ms',
        type=float,
        default=5.0,
        help='Bin width (ms) used to build mutual information matrices in both scripts.',
    )
    parser.add_argument(
        '--mi-lag-ms',
        type=float,
        default=10.0,
        help='Lag window (ms) used when network_analysis recomputes MI from saved spikes.',
    )
    parser.add_argument(
        '--distance-based-mi-h',
        type=int,
        default=10,
        help='Neighborhood size used by the distance-based MI metric.',
    )
    parser.add_argument(
        '--duration-ms',
        type=float,
        default=2000.0,
        help='Simulation duration in milliseconds passed to the underlying scripts.',
    )
    parser.add_argument(
        '--r-ee',
        type=float,
        default=2.0,
        help='R_ee metadata value recorded by network analysis output.',
    )
    parser.add_argument(
        '--louvain-resolution',
        type=float,
        default=1.0,
        help='Resolution parameter for Louvain community detection in network_analysis.py.',
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Run-specific output root. Public figures go in <output-dir>/public and intermediate files go in <output-dir>/internal.',
    )
    args = parser.parse_args()
    if args.mi_bin_width_ms <= 0.0:
        raise ValueError('--mi-bin-width-ms must be positive.')
    if args.mi_lag_ms < 0.0:
        raise ValueError('--mi-lag-ms must be non-negative.')
    if args.distance_based_mi_h <= 0:
        raise ValueError('--distance-based-mi-h must be positive.')
    if args.duration_ms <= 0.0:
        raise ValueError('--duration-ms must be positive.')
    main(
        radius_um=args.radius_um,
        layer_weight_decay_lambda=args.layer_weight_decay_lambda,
        summary_signal=args.summary_signal,
        summary_dt_ms=args.summary_dt_ms,
        mi_bin_width_ms=args.mi_bin_width_ms,
        mi_lag_ms=args.mi_lag_ms,
        distance_based_mi_h=args.distance_based_mi_h,
        duration_ms=args.duration_ms,
        r_ee=args.r_ee,
        louvain_resolution=args.louvain_resolution,
        output_dir=args.output_dir,
    )
