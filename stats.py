from file import *
from collections import Counter
from statistics import mean
from itertools import islice
import itertools
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd


# get duration of each event in a data group (ex. {'A1':[0.26132, 0.13542,...]})
def get_duration(data_group):
    duration = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        sampling_freq = 20000
        x=data[2]
        x[x==0.0] = -1                                                          # replace 0 with -1
        zero_crossings = np.where(np.diff(np.signbit(x)))[0]                    # check when 1 switches to -1 and vice versa
        duration_subgroup = []
        
        for i in range(0,len(zero_crossings),2):
            duration_subgroup.append(zero_crossings[i+1]-zero_crossings[i])   
        duration_subgroup = [i/sampling_freq for i in duration_subgroup]        # duration in seconds
        duration[data_subgroup] = duration_subgroup
    return duration


# returns number of 0's and 1's in a data group (ex. {'A1':[31243, 4253]})
def get_count(data_group):
    count = {}   # in the form of {'A1':[zeros, ones]}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        count[data_subgroup] = [Counter(data[2])[0], Counter(data[2])[1]]
    return count


# returns number of events in a data group (ex. A={'A1':24,...})
def get_groups(data_group):
    group_tot = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        x = data[2]
        group_nr = len(list(itertools.groupby(x, lambda x: x == 1)))
        group_nr = math.ceil((group_nr-1)/2)
        group_tot[data_subgroup] = group_nr
    return group_tot


# returns the mean current of each event (ex. {'A':[3321,2341,42351,...]}
# def get_current(data_group):
#     current = {}
#     temp = []
#     indices = get_index_sign(data_group)
#     for data_subgroup in data_group:
#         data = get_file(data_subgroup)
#         for event in indices[data_subgroup]:
#             temp.append(mean(data[1][event[0]:event[1]+1]))   # range excludes last value so we add +1 back
#         current[data_subgroup] = temp
#         temp = []
#     return current


# returns the mean current of each event (ex. {'A':[3321,2341,42351,...]}
def get_current(data_group):
    current = {}
    temp = []
    indices = get_index_sign(data_group)
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        for event in indices[data_subgroup]:
            temp.append(mean(data[1][event[0]:event[1]+1]))   # range excludes last value so we add +1 back
        current[data_subgroup] = temp
        temp = []
    return current


# returns the mean current of each group (ex. {'A':[3321,2341,42351,...]}
def get_current_all(data_group):
    current = {}
    temp = []
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        temp.append(mean(data[1]))   
        current[data_subgroup] = temp
        temp = []
    return current


# returns the index of events by checking for sign changes (ex. {'A1' = [[start,end],[start,end],...]})
def get_index_sign(data_group):
    indices = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        if data[2][0] == 1 or data[2][-1] == 1:
            print('ERROR: Data begins with a 1 or ends with a 1. Algorithm does not account for this.')
            return -1

        temp = []
        start = np.NaN
        end = np.NaN
        x = data[2]
        x[x==0.0] = -1                                         # replace 0 with -1
        zero_crossings = np.where(np.diff(np.signbit(x)))[0]   # find sign changes from -1 to 1 or vice versa
        
        # check sign cahnges and if it represents a 01 change or a 10 change
        for i in range(len(zero_crossings)-1):
            if data[2][zero_crossings[i]] == 0 and data[2][zero_crossings[i]+1] == 1:   # start: 01
                start = zero_crossings[i]+1     # np.diff returns the the index at 000(0)1111; increment it to get 0000(1)111
                if data[2][zero_crossings[i+1]] == 1 and data[2][zero_crossings[i+1]+1] == 0:   # end: 10
                    end = zero_crossings[i+1]
                    temp.append([start,end])
            start = np.NaN
            end = np.NaN
        indices[data_subgroup] = temp
    return indices


# returns the index of event using sequal differences (ex. {'A1' = [[start,end],[start,end],...]})
def get_index_seqdiff(data_group):
    indices = {}
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        if data[2][0] == 1 or data[2][-1] == 1:
            print('ERROR: Data begins with a 1 or ends with a 1. Algorithm does not account for this.')
            return -1

        temp = []
        start = np.NaN
        end = np.NaN
        x = data[2]
        seqdiff = (x[1:] - x[:-1]).nonzero()
        
        # check if it represents a 01 change or a 10 change
        for i in range(len(seqdiff)-1):
            if data[2][seqdiff[i]] == 0 and data[2][seqdiff[i]+1] == 1:   # start: 01
                start = seqdiff[i]+1     # nonzero() returns the the index at 000(0)1111; increment it to get 0000(1)111
                if data[2][seqdiff[i+1]] == 1 and data[2][seqdiff[i+1]+1] == 0:   # end: 10
                    end = seqdiff[i+1]
                    temp.append([start,end])
            start = np.NaN
            end = np.NaN
        indices[data_subgroup] = temp
    return indices
        
# check if we are rapidly increasing and decreasing; returns {'A' = [[start, end], [start, end], ...]}        
def get_predictions(data_group):
    predictions = {}
    window_size = 100   # in rows
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        temp = []
        sliding_area = len(data[1]) - window_size
        skip_values_i = 0
        # for i in range(skip_values_i, len(data[1]) - window_size):   # check if we're increasing
        while skip_values_i < sliding_area:
            sliding_window = data[1][skip_values_i:skip_values_i+window_size]
            is_increasing = all(p < q for p, q in zip(sliding_window, sliding_window[1:]))
            if is_increasing is True:
                skip_values_j = skip_values_i
                # for j in range(skip_values_j, len(data[1] - window_size)):   # if we're increasing, find the following decreasing
                while skip_values_j < sliding_area:
                    sliding_window = data[1][skip_values_j:skip_values_j+window_size]
                    is_decreasing = all(p > q for p, q in zip(sliding_window, sliding_window[1:]))
                    if is_decreasing is True:
                        temp.append([skip_values_i, skip_values_j])
                        skip_values_i = skip_values_j - 1
                        skip_values_j = sliding_area
                    skip_values_j += 1
            skip_values_i += 1
        predictions[data_subgroup] = temp
        print('Prediction: ', data_subgroup)
    return predictions


# returns the increasing and decreasing slopes of the signal; length of the slope defined in window_shape
def get_slopes(seq):
    window_list = get_window(seq)
    window_size = 25   # defined in [pA]
    slopes = []   # of the form [[increase, decrease], [increase, decrease], ...]
    i = 0
    while i < len(window_list):
        # find the next increase
        if all(p < q for p, q in zip(window_list[i], window_list[i][1:])) is True:
            # and math.fabs(window_list[i][-1] - window_list[i][0]) < window_size+3\
            # and math.fabs(window_list[i][-1] - window_list[i][0]) > window_size-3:
            j = i
            # find the decrease and match it with the previous increase
            while j < len(window_list):
                if all(p > q for p, q in zip(window_list[j], window_list[j][1:])) is True:
                    # and math.fabs(window_list[i][-1] - window_list[i][0]) < window_size+3\
                    # and math.fabs(window_list[i][-1] - window_list[i][0]) > window_size-3:
                    slopes.append([i,j])
                    i = j - 1
                    j = len(window_list)
                j += 1
        i += 1
    return slopes


# returns an array of sliced into windows [[1,2,3],[2,3,4],[3,4,5], ...]
def get_window(seq, window_shape = 20000, step = 2000):
    window = sliding_window_view(seq, window_shape)[::step,:]   # change window size through window_shape
    return window


# def check_slope(data_group):
#     for data_subgroup in data_group:
#         data = get_file(data_subgroup)

#         x=data[2]
#         x[x==0.0] = -1  
#         zero_crossings = np.where(np.diff(np.signbit(x)))[0]    # check where value changes from postive to negative, and vice versa
#         sample_freq = 20000
#         y = [x/sample_freq for x in data[0]]

#         for i in range(0,len(zero_crossings),2):
#             arr = data[1][zero_crossings[i]-300:zero_crossings[i]]
#             if all(p < q for p, q in zip(arr, arr[1:])) is True:
#                 print('increase')
#             else:
#                 print('ERROR increase')
#             arr = data[1][zero_crossings[i+1]:zero_crossings[i+1]+300]
#             if all(p > q for p, q in zip(arr, arr[1:])) is True:
#                 print('decrease')
#             else:
#                 print('ERROR decrease')

# check_slope(data_A)

def get_basic_info(filepath):
    df = pd.read_csv(filepath)
    dfC = pd.read_csv('/Users/valija/Desktop/Master Thesis/code/WINES-PRED-C/C.csv')

    events = np.array(df['Event'])
    seqdiff = (events[1:] - events[:-1]).nonzero()[0]
    start_points = seqdiff[::2]
    end_points = seqdiff[1::2]
    duration = end_points - start_points
    current = dfC['Value']

    print('Nr. of events: ', seqdiff/2)
    print('Avg. event duration: ', np.mean(duration)/20000,' s')
    print('Avg. current: ', np.mean(current),' pA')
    print('Nr. of labels: ', Counter(df['Event']))

# print('Predictions of A in C:\n')
# get_basic_info('/Users/valija/Desktop/Master Thesis/code/WINES-PRED-C/predvidene_C_A.csv')
# print('Predictions of B in C:\n')
# get_basic_info('/Users/valija/Desktop/Master Thesis/code/WINES-PRED-C/predvidene_C_B.csv')
