import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd, label):
    print(f"\n[{label}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(radius_um, layer_weight_decay_lambda, summary_signal, summary_dt_ms, mi_bin_width_ms):
    repo_root = Path(__file__).resolve().parent
    python_exec = sys.executable

    print(f"Using Python interpreter: {python_exec}")
    print(f"Repository root: {repo_root}")
    print("Regenerating all figures with fresh multilayer internal data...")

    run_step(
        [
            python_exec,
            str(repo_root / 'multilayer.py'),
            '--no-show',
            '--summary-signal',
            summary_signal,
            '--summary-dt-ms',
            str(summary_dt_ms),
            '--mi-bin-width-ms',
            str(mi_bin_width_ms),
        ],
        '1/2 multilayer',
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
        ],
        '2/2 neuropil',
    )

    print("\nDone.")
    print("Fresh outputs were generated in output/public and output/internal.")


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
    args = parser.parse_args()
    if args.mi_bin_width_ms <= 0.0:
        raise ValueError('--mi-bin-width-ms must be positive.')
    main(
        radius_um=args.radius_um,
        layer_weight_decay_lambda=args.layer_weight_decay_lambda,
        summary_signal=args.summary_signal,
        summary_dt_ms=args.summary_dt_ms,
        mi_bin_width_ms=args.mi_bin_width_ms,
    )
