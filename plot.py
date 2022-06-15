"""Plotting tools for time series.
"""
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import datashader as ds
import pandas as pd
import colorcet as cc

from mat_file_loader import get_file
from stats import get_duration, get_slopes
import main


def getplot_separate():
    """
    Plots the time series and marks the events.
    """

    # get data
    dfA = pd.read_csv(main.PATH + 'A.csv')
    dfB = pd.read_csv(main.PATH + 'B.csv')
    dfC1 = pd.read_csv(main.PATH + 'C1.csv')
    dfC2 = pd.read_csv(main.PATH + 'C2.csv')
    dfC3 = pd.read_csv(main.PATH + 'C3.csv')

    # add header names if not already added
    dfA = pd.DataFrame(dfA.values, columns=['Time', 'Value', 'Label'])
    dfB = pd.DataFrame(dfB.values, columns=['Time', 'Value', 'Label'])
    dfC1 = pd.DataFrame(dfC1.values, columns=['Time', 'Value'])
    dfC2 = pd.DataFrame(dfC2.values, columns=['Time', 'Value'])
    dfC3 = pd.DataFrame(dfC3.values, columns=['Time', 'Value'])

    # set current plotting data ['A','B','C1','C2','C3']
    df = dfA
    dfname = 'A'

    # plot the data
    if dfname is 'A' or dfname is 'B':
        plt.scatter(
            df['Time']/20000, df.loc[df['Label'] == 0]['Value'], s=0.1,
            c='blue', label='no event')
        plt.scatter(
            df['Time']/20000, df.loc[df['Label'] == 1]['Value'], s=0.1,
            c='red', label='event ' + dfname)
    if dfname is 'C1' or dfname is 'C2' or dfname is 'C3':
        plt.scatter(
            df['Time']/20000, df['Value'], s=0.1, c='blue', label='unlabeled')
    plt.ylabel('Current [pA]')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title(dfname)
    plt.savefig(main.PATH + dfname + '.png')
    plt.close()


def getplot_shade(data_group, group_name):
    """Plots the time series with intensity on more concentrated places.

    Args:
        data_group (h5py.Dataset): The dataset to plot.
        group_name (str): Name of the dataset to be used for saving filenames.
    """
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        sample_freq = 20000
        type_one_index = np.where(data[2] == 1)[0]
        type_one_value = np.copy(data[1][type_one_index, ])
        type_zero_index = np.where(data[2] == 0)[0]
        type_zero_value = np.copy(data[1][type_zero_index, ])
        type_one_index = [x/sample_freq for x in type_one_index]
        type_zero_index = [x/sample_freq for x in type_zero_index]

        ones_new = np.stack((type_one_index, type_one_value), axis=1)
        df = pd.DataFrame(data=ones_new, columns=["x", "y"])
        cvs = ds.Canvas(plot_width=500, plot_height=500)
        agg = cvs.points(df, 'x', 'y')
        img = ds.tf.set_background(
            ds.tf.shade(agg, how="log", cmap=cc.fire), "black").to_pil()

        plt.imshow(img)
        plt.savefig(
            group_name + '_shade.png', facecolor='white', transparent=False)
        plt.close()


def getplot(data_group, group_name):
    """Plots the time series withou marking the events.

    Args:
        data_group (h5py.Dataset): The dataset to plot.
        group_name (str): Name of the dataset to be used for saving filenames.
    """
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        plt.plot(
            data[0], data[1], color='b', linestyle='none', marker='.',
            markersize=1)
        plt.ticklabel_format(style='plain')
        plt.title(data_subgroup)
        plt.ylabel('current [pA]')
        plt.savefig(group_name + '.png', facecolor='white', transparent=False)
        plt.close()


def getplot_bar(data_group, group_name):
    """Plots the histogram of the currents.

    Args:
        data_group (h5py.Dataset): The dataset to plot.
        group_name (str): Name of the dataset to be used for saving filenames.
    """
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        plt.hist(data[1], histtype='bar', ec='black')
        plt.title(data_subgroup)
        plt.xlabel('current [pA]')
        plt.savefig(
            group_name + '_bar.png', facecolor='white', transparent=False)
        plt.close()


def getplot_spec(data_group, group_name):
    """Plot only events, with some non-events on the left and right.

    Args:
        data_group (h5py.Dataset): The dataset to plot.
        group_name (str): Name of the dataset to be used for saving filenames.
    """
    duration = get_duration(data_group)
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        x = data[2]
        x[x == 0.0] = -1
        zero_crossings = np.where(np.diff(np.signbit(x)))[0]
        sample_freq = 20000
        y = [x/sample_freq for x in data[0]]

        for i in range(0, len(zero_crossings), 2):
            # left and right margin of event. Higher value narrower plot
            LR_SIZE = 7
            padding = math.floor(
                (zero_crossings[i+1]-zero_crossings[i])/LR_SIZE)

            plt.scatter(
                y[zero_crossings[i]-padding:zero_crossings[i]],
                data[1][zero_crossings[i]-padding:zero_crossings[i]],
                color='b', s=1)
            plt.scatter(
                y[zero_crossings[i+1]:zero_crossings[i+1]+padding],
                data[1][zero_crossings[i+1]:zero_crossings[i+1]+padding],
                color='b', s=1, label='0')
            plt.scatter(
                y[zero_crossings[i]:zero_crossings[i+1]],
                data[1][zero_crossings[i]:zero_crossings[i+1]], color='r', s=1,
                label='1')
            plt.ticklabel_format(style='plain')
            plt.ylabel('current [pA]')
            plt.xlabel('time [s]')
            plt.grid()
            plt.title(
                data_subgroup + '-' + str(math.ceil(i/2)) + ': ' +
                str(duration[data_subgroup][math.ceil(i/2)]) + 's')
            plt.legend(loc='lower right')
            Path(group_name + '/' + data_subgroup + '_spec/').mkdir(
                parents=True, exist_ok=True)
            plt.savefig(
                group_name + '/' + data_subgroup + '_spec/' + data_subgroup +
                '-' + str(math.ceil(i/2)) + '_spec.png', facecolor='white',
                transparent=False)
            plt.close()
            print("done " + data_subgroup)


def getplot_prediction(data_group, group_name):
    """Plot the predictions.

    Args:
        data_group (h5py.Dataset): The dataset to plot.
        group_name (str): Name of the dataset to be used for saving filenames.
    """
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        slopes = get_slopes(data[1])

        plt.scatter(data[0], data[1], color='b', s=1, label='0')
        for i, _ in enumerate(slopes):
            plt.scatter(
                data[0][slopes[i][0]:slopes[i][1]],
                data[1][slopes[i][0]:slopes[i][1]], color='r', s=1)
        plt.ticklabel_format(style='plain')
        plt.ylabel('current [pA]')
        plt.xlabel('time [s]')
        plt.grid()
        plt.title(data_subgroup)
        plt.legend(loc='lower right')
        plt.savefig(
            group_name + '/' + data_subgroup + '_pred.png', facecolor='white',
            transparent=False)
        plt.close()
        print(group_name + '/' + data_subgroup + '_pred.png')


def getplot_bar_time(data_A, data_B):
    """Plots 2 pages (whole data, 4s, 2s) and (1s, 0.5s, 0.2s) of
    histograms for time distribution of datasets.

    Args:
        data_A (pd.DataFrame): First dataset to be plotted.
        data_B (pd.DataFrame): Second dataset to be plotted.
    """
    # get length of each event
    dataA = get_duration(data_A)
    dataB = get_duration(data_B)
    listA = [j for i in list(dataA.values()) for j in i]
    listB = [j for i in list(dataB.values()) for j in i]

    # set the plotting environment for the first page
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(15, 20)
    fig.suptitle(
        'Time distribution of events in dataset A and B', fontsize='xx-large')

    # histogram of whole data A and data B
    yA, _, barsA = axs[0, 0].hist(
                 listA, histtype='bar', ec='black', bins=20,
                 range=[0, max(listA+listB)])
    yB, _, barsB = axs[0, 1].hist(
                 listB, histtype='bar', ec='black', bins=20,
                 range=[0, max(listA+listB)], color='orange')
    axs[0, 0].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), title='Data A')
    axs[0, 0].bar_label(barsA)
    axs[0, 1].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), title='Data B')
    axs[0, 1].bar_label(barsB)

    # histogram of fist 4 seconds of data A and data B
    yA, _, barsA = axs[1, 0].hist(
                 listA, histtype='bar', ec='black', bins=200,
                 range=[0, max(listA+listB)])
    yB, _, barsB = axs[1, 1].hist(
                 listB, histtype='bar', ec='black', bins=200,
                 range=[0, max(listA+listB)], color='orange')
    axs[1, 0].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 4),
                  title='Data A: zoom in to 4s')
    axs[1, 0].bar_label(barsA)
    axs[1, 1].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 4),
                  title='Data B: zoom in to 4s')
    axs[1, 1].bar_label(barsB)

    # histogram of fist 2 seconds of data A and data B
    yA, _, barsA = axs[2, 0].hist(
                 listA, histtype='bar', ec='black', bins=400,
                 range=[0, max(listA+listB)])
    yB, _, barsB = axs[2, 1].hist(
                 listB, histtype='bar', ec='black', bins=400,
                 range=[0, max(listA+listB)], color='orange')
    axs[2, 0].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 2),
                  title='Data A: zoom in to 2s')
    axs[2, 0].bar_label(barsA)
    axs[2, 1].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 2),
                  title='Data B: zoom in to 2s')
    axs[2, 1].bar_label(barsB)

    # save first page
    plt.savefig(
        main.PATH + 'AB_bar-1.png', facecolor='white', transparent=False)
    plt.close()

    # set the plotting environment for the second page
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(15, 20)
    fig.suptitle(
        'Time distribution of events in dataset A and B', fontsize='xx-large')

    # histogram of fist 1 second of data A and data B
    yA, _, barsA = axs[0, 0].hist(
                 listA, histtype='bar', ec='black', bins=800,
                 range=[0, max(listA+listB)])
    yB, _, barsB = axs[0, 1].hist(
                 listB, histtype='bar', ec='black', bins=800,
                 range=[0, max(listA+listB)], color='orange')
    axs[0, 0].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 1),
                  title='Data A: zoom in to 1s')
    axs[0, 0].bar_label(barsA)
    axs[0, 1].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 1),
                  title='Data B: zoom in to 1s')
    axs[0, 1].bar_label(barsB)

    # histogram of fist 0.5 seconds of data A and data B
    yA, _, barsA = axs[1, 0].hist(
                 listA, histtype='bar', ec='black', bins=1600,
                 range=[0, max(listA+listB)])
    yB, _, barsB = axs[1, 1].hist(
                 listB, histtype='bar', ec='black', bins=1600,
                 range=[0, max(listA+listB)], color='orange')
    axs[1, 0].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 0.5),
                  title='Data A: zoom in to 0.5s')
    axs[1, 0].bar_label(barsA)
    axs[1, 1].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 0.5),
                  title='Data B: zoom in to 0.5s')
    axs[1, 1].bar_label(barsB)

    # histogram of fist 0.2 seconds of data A and data B
    yA, _, barsA = axs[2, 0].hist(
                 listA, histtype='bar', ec='black', bins=3800,
                 range=[0, max(listA+listB)])
    yB, _, barsB = axs[2, 1].hist(
                 listB, histtype='bar', ec='black', bins=3800,
                 range=[0, max(listA+listB)], color='orange')
    axs[2, 0].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 0.2),
                  title='Data A: zoom in to 0.2s')
    axs[2, 0].bar_label(barsA)
    axs[2, 1].set(xlabel='time [s]', ylabel='nr. of events',
                  ylim=(0, max(yA.tolist()+yB.tolist())+10), xlim=(0, 0.2),
                  title='Data B: zoom in to 0.2s')
    axs[2, 1].bar_label(barsB)

    # save second page
    plt.savefig(
        main.PATH + 'AB_bar-2.png', facecolor='white', transparent=False)
    plt.close()
