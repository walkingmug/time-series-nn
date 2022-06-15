"""Window manipulation tools.
"""
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import numpy as np

import main


def get_window(seq, window_shape=main.WINDOW_SIZE, step=main.WINDOW_STEP):
    """Splits a series into windows.

    Args:
        seq (list): Array to be sliced.
        window_shape (int, optional): Size of the window in rows.
        step (int, optional): Sliding step for the window.

    Returns:
        numpy.ndarray: Series sliced into windows.
    """
    window = sliding_window_view(seq, window_shape)[::step, :]

    return window


def get_labeled_windows(
        dataset, unlabeled_winds, window_size=main.WINDOW_SIZE,
        window_step=main.WINDOW_STEP, window_thresh=main.WINDOW_THRESH,
        use_regression_mean=False):
    """Gets the windows and labels them with a class.

    Args:
        dataset (pandas.DataFrame): Labeled dataset with columns
        ['Time', 'Value', 'Label']
        unlabeled_winds (list): Unlabeled dataset with columns
        ['Time', 'Value']. Must contain pandas.DataFrame types.
        window_size (int): Size of windows to be generated.
        window_step (int): The step to slide the window.
        window_thresh (float): At which class quantity level to assign the same
        class.
        use_regression_mean (bool, optional): Takes the arithmetic mean value.
        of windows (Logistic Regression). Defaults to False.

    Returns:
        pandas.DataFrame: The labeled windows.
        list: Contains windows of unlabeled dataset of type pandas.DataFrame.
    """
    # split data into windows
    x_window = get_window(
             dataset['Value'], window_shape=window_size, step=window_step)
    y_window = get_window(
             dataset['Label'], window_shape=window_size, step=window_step)
    unlabeled_wind = []
    for dframe in unlabeled_winds:
        unlabeled_wind.append(
            pd.DataFrame(np.array(get_window(
                dframe['Value'], window_shape=window_size, step=window_step))))

    # label windows with 0, 1 or 2 based on the quantinty of a class present
    # if Logistic Regression is used, it takes the arithmetic mean of windows
    values_and_labels = []
    if use_regression_mean is True:
        for i, _ in enumerate(y_window):
            if (np.count_nonzero(y_window[i] == 2) >=
                (window_thresh * len(y_window[i]))):
                values_and_labels.append([float(np.mean(x_window[i]))])
                values_and_labels[-1].append(2)
            elif (np.count_nonzero(y_window[i] == 1) >=
                  (window_thresh * len(y_window[i]))):
                values_and_labels.append([float(np.mean(x_window[i]))])
                values_and_labels[-1].append(1)
            else:
                values_and_labels.append([float(np.mean(x_window[i]))])
                values_and_labels[-1].append(0)
        new_unlabeled_wind = []
        for dframe in unlabeled_wind:
            temp = []
            for _, val in enumerate(np.array(dframe)):
                temp.append(float(np.mean(val)))
            new_unlabeled_wind.append(pd.DataFrame(np.array(temp)))
        unlabeled_wind = new_unlabeled_wind
    else:
        for i, _ in enumerate(y_window):
            if (np.count_nonzero(y_window[i] == 2) >=
                (window_thresh * len(y_window[i]))):
                values_and_labels.append(list(x_window[i]))
                values_and_labels[-1].append(2)
            elif (np.count_nonzero(y_window[i] == 1) >=
                  (window_thresh * len(y_window[i]))):
                values_and_labels.append(list(x_window[i]))
                values_and_labels[-1].append(1)
            else:
                values_and_labels.append(list(x_window[i]))
                values_and_labels[-1].append(0)
    labeled_wind = pd.DataFrame(np.array(values_and_labels))
    labeled_wind.columns = [*labeled_wind.columns[:-1], 'Label']

    return labeled_wind, unlabeled_wind


def predicted_windows_to_rows(df, predicted_label):
    """Matches the windows and their predicted classes with the original rows.
    Saves the results.

    Args:
        df (pandas.DataFrame): _description_
        predicted_label (_type_): _description_
    """
    app = np.zeros(len(df['Time']), dtype=int)
    for index, label in enumerate(predicted_label['Value']):
        app[index * 200:index * 200 + 2001] = label
    df['Label'] = app
    df['Time'] = range(1, len(df['Time']) + 1)
    df.to_csv(main.PATH + 'C1.csv', index=False)
