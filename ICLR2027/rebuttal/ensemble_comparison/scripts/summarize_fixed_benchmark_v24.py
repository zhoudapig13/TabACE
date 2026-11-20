"""Collect all completed fixed-benchmark reference-strength conditions."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'results'
DEST = ROOT / 'paperish_fixed_benchmark_v24'
SOURCES = [DEST, ROOT / 'paperish_fixed_benchmark_reference020_v25']


def main():
    frames = [pd.read_csv(p) for root in SOURCES for p in sorted(root.glob('averages_*.csv'))]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(DEST / 'all_completed_averages.csv', index=False)
    parts = [
        '# Frozen-benchmark Shared / Specific replay',
        '',
        'Benchmark inputs and Base predictions are held fixed. Seed = 0. '
        'Evaluation strengths are {0.2, 0.5, 0.8, 1.0}. '
        'Classification uses 72 datasets for Clean/five corruptions and 35 for seven shifts; '
        'regression uses 35 and 33, respectively.',
        '',
        'Observed-only uses no clean reference rows. Reference005 and reference020 '
        'replace at most 5% and 20% of the rows within each Shared view using clean '
        'FT TRAIN examples disjoint from calibration and final evaluation rows. '
        'This is additional clean-reference supervision for the frozen predictor. '
        'All context sizes remain at most 256. Shared has five alternative views; '
        'Specific inherits them and adds five recovered family-directed views. '
        'The backbone stays frozen and the original fusion gate is used.',
        '',
        'Fusion policies are fitted with five-fold dataset-level OOF calibration '
        '(available calibration strengths: clean, 0.5 and 1.0). '
        'The reference fractions are fixed conditions, not selected using final query scores. '
        'All completed conditions are reported.',
        '',
        'This is a controlled reconstruction on the frozen Table 28 development benchmark, '
        'rather than exact archived main-table replication. Regression Clean/five '
        'remain the existing semi-synthetic tasks. The original complete deployed '
        'Specific lists have not been recovered; this run uses ten reconstructed '
        'alternative views. Prior adapted Table 28 and rebuttal/PDF files are preserved.',
    ]
    for variant in ['observed_only', 'reference005', 'reference020']:
        subset = frame[frame.variant == variant]
        if subset.empty:
            continue
        parts += ['', f'## {variant}', '', '| Task | Method | Clean | Five corruptions | Seven shifts |',
                  '|---|---|---|---|---|']
        for task in ['classification', 'regression']:
            a, b = ('accuracy', 'auc') if task == 'classification' else ('r2', 'nrmse')
            for method in ['Base', 'Shared', 'Specific']:
                cells = []
                for block in ['clean', 'five', 'seven']:
                    row = subset[(subset.task == task) & (subset.method == method) & (subset.block == block)]
                    cells.append('Pending' if row.empty else f'{row.iloc[0][a]:.4f} / {row.iloc[0][b]:.4f}')
                parts.append('| ' + ' | '.join([task, method] + cells) + ' |')
    (DEST / 'RESULTS.md').write_text('\n'.join(parts) + '\n')
    print(DEST / 'RESULTS.md')


if __name__ == '__main__':
    main()
