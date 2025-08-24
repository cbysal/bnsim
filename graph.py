import bisect
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def sample_indices(x_data, n=20):
    return np.linspace(0, max(x_data), n)

plt.rcParams['font.family'] = 'Times New Roman'
if sys.argv[1] == "correction-factor":
    df = pd.read_csv(sys.argv[2], header=None)
    with_errbar = len(sys.argv) == 5

    df.iloc[:, :13] *= 100
    y_data = df[0]
    y1_data = df.iloc[:, 1:13].mean(axis=1)
    y2_data = df.iloc[:, 13:25].mean(axis=1)
    y1_err = df.iloc[:, 1:13].sub(y1_data, axis=0).pow(2).sum(axis=1).pow(0.5)
    y2_err = df.iloc[:, 13:25].sub(y2_data, axis=0).pow(2).sum(axis=1).pow(0.5)

    fig1, ax11 = plt.subplots(figsize=(5, 3))

    ax11.set_xlabel('Correction Factor/%')
    ax11.set_ylabel('Recommunication Ratio/%')
    ax11.set_xlim(y_data.min(), y_data.max())
    ax11.set_ylim(0, 100)
    ax11.plot(y_data, y1_data, label='Recommunication Ratio/%', color='r')

    ax12 = ax11.twinx()
    ax12.set_ylabel('Time/ms')
    ax12.plot(y_data, y2_data, label='Propagation Time/ms', color='g')

    ax11.grid(True)

    lines1 = ax11.get_lines() + ax12.get_lines()
    labels1 = [line.get_label() for line in lines1]
    ax11.legend(lines1, labels1, loc='upper right')

    zoom_xlim = (115, 120)
    zoom_y1_min, zoom_y1_max = (y1_data[(y_data >= zoom_xlim[0] - 0.01) & (y_data <= zoom_xlim[1] + 0.01)].min(),
                                y1_data[(y_data >= zoom_xlim[0] - 0.01) & (y_data <= zoom_xlim[1] + 0.01)].max())
    zoom_y1_min, zoom_y1_max = zoom_y1_min - (zoom_y1_max - zoom_y1_min) * 0.2, zoom_y1_max + (zoom_y1_max - zoom_y1_min) * 0.2

    zoom_y2_min, zoom_y2_max = (y2_data[(y_data >= zoom_xlim[0] - 0.01) & (y_data <= zoom_xlim[1] + 0.01)].min(),
                                y2_data[(y_data >= zoom_xlim[0] - 0.01) & (y_data <= zoom_xlim[1] + 0.01)].max())
    zoom_y2_min, zoom_y2_max = zoom_y2_min - (zoom_y2_max - zoom_y2_min) * 0.2, zoom_y2_max + (zoom_y2_max - zoom_y2_min) * 0.2

    y1_min, y1_max = ax11.get_ylim()
    y2_min, y2_max = ax12.get_ylim()
    zoom_y1_min, zoom_y2_min = y1_min + (y1_max - y1_min) * min((zoom_y1_min - y1_min) / (y1_max - y1_min), (zoom_y2_min - y2_min) / (y2_max - y2_min)), y2_min + (y2_max - y2_min) * min((zoom_y1_min - y1_min) / (y1_max - y1_min), (zoom_y2_min - y2_min) / (y2_max - y2_min))
    zoom_y1_max, zoom_y2_max = y1_min + (y1_max - y1_min) * max((zoom_y1_max - y1_min) / (y1_max - y1_min), (zoom_y2_max - y2_min) / (y2_max - y2_min)), y2_min + (y2_max - y2_min) * max((zoom_y1_max - y1_min) / (y1_max - y1_min), (zoom_y2_max - y2_min) / (y2_max - y2_min))

    axins = inset_axes(ax11, width=2, height=1.2, loc="right", bbox_to_anchor=(320, 100))
    axins2 = axins.twinx()

    axins.plot(y_data, y1_data, 'r')
    axins2.plot(y_data, y2_data, 'g')

    axins.set_xlim(zoom_xlim)
    axins2.set_xlim(zoom_xlim)

    axins.set_ylim(zoom_y1_min, zoom_y1_max)
    axins2.set_ylim(zoom_y2_min, zoom_y2_max)

    axins.set_yticks([])
    axins2.set_yticks([])

    xticks = [115, 116, 117, 118, 119, 120]
    axins.set_xticks(xticks)

    mark_inset(ax11, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig1.savefig(sys.argv[3], bbox_inches='tight')

if sys.argv[1] == "scalability":
    plt.figure(figsize=(4, 2.4))
    df = pd.read_csv(sys.argv[2], header=None)
    labels = ['Native', 'Alias', 'BCB', 'ECCB']

    for i in range(4):
        x_sample = np.linspace(0, max(df.iloc[:, i * 12: (i + 1) * 12].max()), 20)
        y_sample = []
        for j in range(12):
            column = []
            for x in x_sample:
                col = df.iloc[:, i * 12 + j]
                if x < col.iloc[0]:
                    column.append(0)
                elif x >= col.iloc[-1]:
                    column.append(100)
                else:
                    idx = bisect.bisect_left(col, x) - 1
                    y = idx + (x - col.iloc[idx]) / (col.iloc[idx + 1] - col.iloc[idx])
                    column.append(y / (len(df) - 1) * 100)
            y_sample.append(column)
        y_sample = np.array(y_sample).T
        y_mean = y_sample.mean(axis=1)
        y_max = y_sample.max(axis=1)
        y_min = y_sample.min(axis=1)
        y_err = [abs(y_mean - y_min), abs(y_max - y_mean)]
        plt.errorbar(x_sample, y_mean, yerr=y_err, label=labels[i], capsize=3)

    plt.xlabel('Time/ms')
    plt.ylabel('Reachability/%')
    plt.xlim(0)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(sys.argv[3], bbox_inches='tight')

if sys.argv[1] == 'block-size':
    plt.figure(figsize=(6, 3.6))
    df = pd.read_csv(sys.argv[2], header=None)
    labels = ['Native', 'Alias', 'BCB', 'ECCB']
    for i in range(0, 4):
        x_data = df.iloc[:, i * 13 + 1: (i + 1) * 13]
        y_data = df[i * 13]
        x_data_unrolled = x_data.values.flatten()
        y_data_repeated = y_data.repeat(x_data.shape[1]).values
        scatter = plt.scatter(x_data_unrolled, y_data_repeated, label=labels[i], marker='x', linewidths=1, s=10)

    plt.xlabel('Time/ms')
    plt.ylabel('Block Size/KB')
    plt.xlim(0, 12000)
    plt.ylim(0)
    plt.legend()
    plt.grid(True)
    plt.savefig(sys.argv[3], bbox_inches='tight')

if sys.argv[1] == "bandwidth":
    plt.figure(figsize=(4, 2.4))
    df = pd.read_csv(sys.argv[2], header=None)
    labels = ['Native', 'Alias', 'BCB', 'ECCB']

    for i in range(4):
        x_sample = np.linspace(0, max(df.iloc[:, i * 12: (i + 1) * 12].max()), 20)
        y_sample = []
        for j in range(12):
            column = []
            for x in x_sample:
                col = df.iloc[:, i * 12 + j]
                if x < col.iloc[0]:
                    column.append(0)
                elif x >= col.iloc[-1]:
                    column.append(100)
                else:
                    idx = bisect.bisect_left(col, x) - 1
                    y = idx + (x - col.iloc[idx]) / (col.iloc[idx + 1] - col.iloc[idx])
                    column.append(y / (len(df) - 1) * 100)
            y_sample.append(column)
        y_sample = np.array(y_sample).T
        y_mean = y_sample.mean(axis=1)
        y_max = y_sample.max(axis=1)
        y_min = y_sample.min(axis=1)
        y_err = [abs(y_mean - y_min), abs(y_max - y_mean)]
        plt.errorbar(x_sample, y_mean, yerr=y_err, label=labels[i], capsize=3)

    plt.xlabel('Time/ms')
    plt.ylabel('Reachability/%')
    plt.xlim(0)
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(sys.argv[3], bbox_inches='tight')

if sys.argv[1] == "similarity":
    plt.figure(figsize=(6, 3.6))
    df = pd.read_csv(sys.argv[2], header=None)
    labels = ['Native', 'Alias', 'BCB-60%', 'BCB-70%', 'BCB-80%',
              'BCB-90%', 'ECCB-60%', 'ECCB-70%', 'ECCB-80%', 'ECCB-90%']

    for i in range(0, 10):
        x_sample = np.linspace(0, max(df.iloc[:, i * 12: (i + 1) * 12].max()), 20)
        y_sample = []
        for j in range(12):
            column = []
            for x in x_sample:
                col = df.iloc[:, i * 12 + j]
                if x < col.iloc[0]:
                    column.append(0)
                elif x >= col.iloc[-1]:
                    column.append(100)
                else:
                    idx = bisect.bisect_left(col, x) - 1
                    y = idx + (x - col.iloc[idx]) / (col.iloc[idx + 1] - col.iloc[idx])
                    column.append(y / (len(df) - 1) * 100)
            y_sample.append(column)
        y_sample = np.array(y_sample).T
        y_mean = y_sample.mean(axis=1)
        y_max = y_sample.max(axis=1)
        y_min = y_sample.min(axis=1)
        y_err = [abs(y_mean - y_min), abs(y_max - y_mean)]
        plt.errorbar(x_sample, y_mean, yerr=y_err, label=labels[i], capsize=3)

    plt.xlabel('Time/ms')
    plt.ylabel('Reachability/%')
    plt.xlim(0)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(sys.argv[3], bbox_inches='tight')
