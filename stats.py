"""Tools for generating statistics about the time series.
Some functions require the data to be in MATLAB format.
"""
from collections import Counter
from statistics import mean
import itertools
import math
from sklearn.metrics import roc_auc_score
import numpy as np

from mat_file_loader import get_file
from window_tools import get_window
import main


def get_confidence(
        y_test, y_test_predicted_label, y_test_predicted_prob, threshold=0.7):
    """Gets the indices of highly confident predictions for test set.

    Args:
        y_test (numpy.ndarray): Labels of the test set.
        y_test_predicted_label (numpy.ndarray): Predicted labels for test set.
        y_test_predicted_prob (numpy.ndarray): Predicted probability for test
        set.
        threshold (float, optional): Threshold for confidence. Defaults to 0.7.

    Returns:
        numpy.ndarray: Correct confidence levels.
        numpy.ndarray: Incorrect confidence levels for class 1.
        numpy.ndarray: Incorrect confidence levels for class 2.
    """
    TP, FP, true_confidence, false_confidence1, false_confidence2 = \
        {}, {}, {}, {}, {}
    for i in range(main.NUM_CLASS):
        # get indices of True Positive, False Positive
        TP[i] = np.argwhere(
            (np.array(y_test) == i) &
            (np.array(y_test_predicted_label) == i)).flatten()
        FP[i] = np.argwhere(
            (np.array(y_test) == i) &
            (np.array(y_test_predicted_label) != i)).flatten()
        np.intersect1d(
            TP[i], np.argwhere(y_test_predicted_prob[:, i] >=
                               threshold).flatten())

        # filter those indices using probability threshold
        not_class = [j for j in range(main.NUM_CLASS) if j != i]
        true_confidence[i] = np.intersect1d(
            TP[i], np.argwhere(y_test_predicted_prob[:, i] >=
                               threshold).flatten())
        false_confidence1[i] = np.intersect1d(
            FP[i], np.argwhere(y_test_predicted_prob[:, not_class[0]] >=
                               threshold).flatten())
        false_confidence2[i] = np.intersect1d(
            FP[i], np.argwhere(y_test_predicted_prob[:, not_class[1]] >=
                               threshold).flatten())

    return true_confidence, false_confidence1, false_confidence2


def get_roc_auc(y, y_predicted_probability, method):
    """AUC value.

    Args:
        y (numpy.ndarray): Labels of test set.
        y_predicted_probability (_type_): Probability predictions of test set.
        method (str): 'ovr' or 'ovo'.

    Returns:
        float: AUC score.
    """
    auc_score = roc_auc_score(
        y_true=y, y_score=y_predicted_probability, multi_class=method)

    return round(auc_score, 4)


def get_duration(data_group):
    """Gets duration of each event in a data group.

    Args:
        data_group (list): The group from mat_file_loader.py.

    Returns:
        dict: Duration of each subgroup.
        (ex. {'A1':[0.26132, 0.13542,...]}).
    """
    duration = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        # replace 0 with -1 to check when sign changes occur
        x = data[2]
        x[x == 0.0] = -1
        zero_crossings = np.where(np.diff(np.signbit(x)))[0]

        # take the difference of endpoint and startpoint of events
        duration_subgroup = []
        for i in range(0, len(zero_crossings), 2):
            duration_subgroup.append(zero_crossings[i+1]-zero_crossings[i])
        duration_subgroup = [i/20000 for i in duration_subgroup]
        duration[data_subgroup] = duration_subgroup

    return duration


def get_count(data_group):
    """Gets the quantity of 0's and 1's in a data group.

    Args:
        data_group (list): The group from mat_file_loader.py.

    Returns:
        dict: Quantity of 0's and 1's in each subgroup.
        (ex. {'A1':[31243, 4253]}).
    """
    count = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        count[data_subgroup] = [Counter(data[2])[0], Counter(data[2])[1]]

    return count


def get_groups(data_group):
    """Gets the number of events in a data group.

    Args:
        data_group (list): The group from mat_file_loader.py.

    Returns:
        dict: Quantity of events in each subgroup.
        (ex. A={'A1':24,...}).
    """
    group_tot = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        x = data[2]
        group_nr = len(list(itertools.groupby(x, lambda x: x == 1)))
        group_nr = math.ceil((group_nr-1) / 2)
        group_tot[data_subgroup] = group_nr

    return group_tot


def get_current(data_group):
    """Gets the mean current of events in a data group.

    Args:
        data_group (list): The group from mat_file_loader.py.

    Returns:
        dict: Mean current of events in each subgroup.
        (ex. {'A':[3321,2341,42351,...]}).
    """
    current = {}
    temp = []
    indices = get_index_sign(data_group)  # get start and end points of events
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        for event in indices[data_subgroup]:
            temp.append(mean(data[1][event[0]:event[1] + 1]))
            # range excludes last value so we add +1 back
        current[data_subgroup] = temp
        temp = []

    return current


def get_current_all(data_group):
    """Gets the mean current in a data group.

    Args:
        data_group (list): The group from mat_file_loader.py.

    Returns:
        dict: Mean current of events in each subgroup.
        (ex. {'A':[3321,2341,42351,...]}).
    """
    current = {}
    temp = []
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        temp.append(mean(data[1]))
        current[data_subgroup] = temp
        temp = []

    return current


def get_index_sign(data_group):
    """Gets indices of events using zero crossings.

    Args:
        data_group (list): The group from mat_file_loader.py.

    Returns:
        dict: Mean current of events in each subgroup.
        (ex. {'A1' = [[start,end], [start,end],...]}).
    """
    indices = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        if data[2][0] is 1 or data[2][-1] is 1:
            print('ERROR: Data begins with a 1 or ends with a 1. \
                  Algorithm does not account for this.')
            return -1

        temp = []
        start = np.NaN
        end = np.NaN
        x = data[2]
        x[x == 0.0] = -1  # replace 0 with -1
        # find sign changes from -1 to 1 or vice versa
        zero_crossings = np.where(np.diff(np.signbit(x)))[0]

        # check sign changes and if it represents a 01 change or a 10 change
        for i in range(len(zero_crossings) - 1):
            if (data[2][zero_crossings[i]] is 0 and
                data[2][zero_crossings[i] + 1] is 1):  # start: 01
                # np.diff returns the the index at 000(0)1111;
                # increment it to get 0000(1)111
                start = zero_crossings[i] + 1
                if (data[2][zero_crossings[i+1]] is 1 and
                    data[2][zero_crossings[i+1] + 1] == 0):  # end: 10
                    end = zero_crossings[i+1]
                    temp.append([start, end])
            start = np.NaN
            end = np.NaN
        indices[data_subgroup] = temp

    return indices


def get_index_seqdiff(data_group):
    """Gets indices of events using sequential difference.

    Args:
        data_group (list): The group from mat_file_loader.py.

    Returns:
        dict: Mean current of events in each subgroup.
        (ex. {'A1' = [[start,end], [start,end],...]}).
    """
    indices = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        if data[2][0] is 1 or data[2][-1] is 1:
            print('ERROR: Data begins with a 1 or ends with a 1. \
                  Algorithm does not account for this.')
            return -1

        temp = []
        start = np.NaN
        end = np.NaN
        x = data[2]
        seqdiff = (x[1:] - x[:-1]).nonzero()

        # check if it represents a 01 change or a 10 change
        for i in range(len(seqdiff)-1):
            # start: 01
            if data[2][seqdiff[i]] is 0 and data[2][seqdiff[i]+1] is 1:
                # nonzero() returns the the index at 000(0)1111;
                # increment it to get 0000(1)111
                start = seqdiff[i]+1
                # end: 10
                if data[2][seqdiff[i+1]] is 1 and data[2][seqdiff[i+1]+1] is 0:
                    end = seqdiff[i+1]
                    temp.append([start, end])
            start = np.NaN
            end = np.NaN
        indices[data_subgroup] = temp
    return indices


def get_slopes(seq):
    """Gets the slopes of the time series.
    Length of the slope is defined in window_shape.

    Args:
        seq (list): The sequence to search in.

    Returns:
        list: Increasing and decreasing indices of slopes.
        ([[increase, decrease], [increase, decrease], ...]).
    """
    window_list = get_window(seq)
    slopes = []
    i = 0

    while i < len(window_list):
        # find the next increase
        if all(p < q for p, q in zip(window_list[i], window_list[i][1:])):
            j = i
            # find the decrease and match it with the previous increase
            while j < len(window_list):
                if all(
                    p > q for p, q in zip(window_list[j], window_list[j][1:])):
                    slopes.append([i, j])
                    i = j - 1
                    j = len(window_list)
                j += 1
        i += 1

    return slopes
