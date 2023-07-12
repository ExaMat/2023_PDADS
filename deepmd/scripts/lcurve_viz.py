#!/usr/bin/env python3
""" Visualize the learning progress for a given lcurve.out

Note that it expects an lcruve.out in the current working directory.

TODO make this a little more flexible with the argparse support; but this works
for now.

This creates three plots stacked on one another:

1. rmse_e_val vs. rmse_e_trn and rmse_f_val vs. rmse_f_trn
2. rmse_val vs. rmse_trn
3. learning rate

Usage:

    lcurve_viz.py out_file.pdf
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def usage():
    usage_text = """
    Usage:

    lcurve_viz.py out_file.pdf
    """
    print(usage_text)

if __name__ == '__main__':
    if not Path('lcurve.out').exists():
        print('lcurve.out not found ... exiting')
        usage()
        sys.exit(1)

    if len(sys.argv) != 2:
        print('Did not specify output file image')
        usage()
        sys.exit(2)

    # matplotlib.use('pdf', force=True)

    lcurve = np.genfromtxt("lcurve.out", names=True)

    rmse_e_trn_df = \
        pd.DataFrame({'value': lcurve['rmse_e_trn'],
                      'lr'   : lcurve['lr'],
                      'step' : lcurve['step'],
                      'type' : 'energy',
                      'mode' : 'training'})
    rmse_e_val_df = \
        pd.DataFrame({'value': lcurve['rmse_e_val'],
                      'lr'   : lcurve['lr'],
                      'step' : lcurve['step'],
                      'type' : 'energy',
                      'mode' : 'validation'})
    rmse_f_trn_df = \
        pd.DataFrame({'value': lcurve['rmse_f_trn'],
                      'lr'   : lcurve['lr'],
                      'step' : lcurve['step'],
                      'type' : 'force',
                      'mode' : 'training'})
    rmse_f_val_df = \
        pd.DataFrame({'value': lcurve['rmse_f_val'],
                      'lr'   : lcurve['lr'],
                      'step' : lcurve['step'],
                      'type' : 'force',
                      'mode' : 'validation'})

    training_df = pd.concat(
        [rmse_e_trn_df, rmse_e_val_df, rmse_f_trn_df, rmse_f_val_df],
        ignore_index=True)

    # Want to stack the three plots
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    plot = sns.lineplot(data=training_df[(training_df.step > 10) & (training_df.step % 500 == 0)], x='step', y='value',
                        hue='mode', style='type', ax=axes[0])

    plot.set_title('rmse_e_val vs. rmse_e_trn and rmse_f_val vs. rmse_f_trn')

    rmse_trn_df = pd.DataFrame({'value': lcurve['rmse_trn'],
                                'step' : lcurve['step'],
                                'mode' : 'training'})

    rmse_val_df = pd.DataFrame({'value': lcurve['rmse_val'],
                                'step' : lcurve['step'],
                                'mode' : 'validation'})

    rmse_df = pd.concat([rmse_trn_df, rmse_val_df], ignore_index=True)

    trn_vs_val_plot = sns.lineplot(data=rmse_df[(rmse_df.step > 10) & (rmse_df.step % 500 == 0)], x='step',
                                   y='value', hue='mode', style='mode', ax=axes[1])

    trn_vs_val_plot.set_title('rmse_val vs. rmse_trn')

    lr_plot = sns.lineplot(data=training_df[(training_df.step > 10) & (training_df.step % 500 == 0)], x='step', y='lr',
                           ax=axes[2])

    lr_plot.set_title('learning rate')

    fig.tight_layout()

    fig.savefig(sys.argv[1])
