"""Collects the data and pre-processes it for training. Predicts the dataset.
"""
import sys

from collections import Counter
from sklearn.utils import class_weight

import pandas as pd
import numpy as np
import tensorflow as tf

from classifiers import (
                         model_dense, model_random_forest, model_lstm,
                         model_logistic_regression, predict_using_model)
from window_tools import get_labeled_windows
from stats import get_roc_auc


# used to replicate results in Sequential model
np.random.seed(1234)
tf.random.set_seed(1234)

# initialize window
SECONDS = 0.2
WINDOW_SIZE = int(SECONDS*20000)
WINDOW_STEP = int(SECONDS*2000)
WINDOW_THRESH = 0.8

# change hyperparameters
NUM_CLASS = 3
SPLIT_TRAIN_TEST = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 11

# get data
df_A = pd.read_csv(filepath_or_buffer='A.csv', delimiter=',')
df_B = pd.read_csv(filepath_or_buffer='B.csv', delimiter=',')
df_C1 = pd.read_csv(filepath_or_buffer='C1.csv', delimiter=',')
df_C2 = pd.read_csv(filepath_or_buffer='C2.csv', delimiter=',')
df_C3 = pd.read_csv(filepath_or_buffer='C3.csv', delimiter=',')

# drop the time column
df_A.drop(['Time'], inplace=True, axis=1)
df_B.drop(['Time'], inplace=True, axis=1)
df_C = [df_C1, df_C2, df_C3]
for dframe in df_C:
    dframe.drop(['Time'], inplace=True, axis=1)

# change class names in B and join it with A to create single training set
df_B['Label'] = df_B['Label'].replace(1, 2)
df_AB = pd.concat([df_A, df_B], ignore_index=True, sort=False)

# working PATH
PATH = ''

# set training set and give it a name (for saving figures, etc.)
dataset = df_AB
DATASET_NAME = 'AB'

# save results with these column names
col = ['algorithm', 'use weight', 'window size', 'window step',
       'window thresh', 'Train: no event', 'Train: event A', 'Train: event B',
       'Test: no event', 'Test: event A', 'Test: event B', 'Pred C1: no event',
       'Pred C1: event A', 'Pred C1: event B', 'Pred C2: no event',
       'Pred C2: event A', 'Pred C2: event B', 'Pred C3: no event',
       'Pred C3: event A', 'Pred C3: event B', 'AUC (ovr)', 'AUC (ovo)']
df_result = pd.DataFrame(columns=col)

# set the parameters and algorithms for training (runs in iteration)
seconds_list = [0.1, 0.2, 0.5, 1, 2, 5, 10]
class_weight_list = [True, False]
algorithm_list = ['LSTM', 'Logistic_Regression', 'Random_Forest', 'Dense']

# train, test and predict the time series
for algorithm in algorithm_list:
    for use_weight in class_weight_list:
        for second in seconds_list:

            # set window size by converting seconds into rows
            WINDOW_SIZE = int(second*20000)
            WINDOW_STEP = int(second*2000)

            # threshold checks if a certain amount of classes are in a window
            # to apply that class. Bigger window means smaller threshold
            if second == 10:
                WINDOW_THRESH = 0.1
            elif second == 5:
                WINDOW_THRESH = 0.4
            elif second == 2:
                WINDOW_THRESH = 0.7
            else:
                WINDOW_THRESH = 0.8

            # some parameters require changing depending on the algorithm
            # Logistic Regression requires arithmetic mean
            USE_MEAN = True if algorithm == 'Logistic_Regression' else False
            # specific requirement to output probability
            USE_PROBA = False if algorithm == 'Dense' or algorithm == 'LSTM' \
                else True
            # it is not beneficial to shuffle on LSTM
            USE_SHUFFLE = False if algorithm == 'LSTM' else True
            # LSTM requires specific input shape
            USE_RESHAPE = True if algorithm == 'LSTM' else False

            # used for progress
            print('\nAlgorithm:', algorithm, '| Use weight:', use_weight,
                  '| Window size:', WINDOW_SIZE/20000, 's | Window step:',
                  WINDOW_STEP/20000, 's | Window thresh.:',
                  WINDOW_THRESH*100, '%')

            # split data into windows and label them
            labeled_wind, unlabeled_wind = get_labeled_windows(
                                         dataset, df_C, WINDOW_SIZE,
                                         WINDOW_STEP, WINDOW_THRESH,
                                         use_regression_mean=USE_MEAN)

            # shuffle windows
            if USE_SHUFFLE is True:
                labeled_wind = labeled_wind.sample(frac=1, random_state=1)

            # # normalizes the data 0..1. Used only for testing purposes
            # labeled_wind = (labeled_wind - labeled_wind.min()) / \
            #              (labeled_wind.max() - labeled_wind.min())
            # labeled_wind['Label'] = labeled_wind['Label'] * 2

            # split data into training set and test set
            if USE_SHUFFLE is False:
                lst = [
                    labeled_wind.loc[labeled_wind['Label'] == 0].iloc[
                        :round(SPLIT_TRAIN_TEST*len(labeled_wind.loc[
                            labeled_wind['Label'] == 0])), :],
                    labeled_wind.loc[labeled_wind['Label'] == 1].iloc[
                        :round(SPLIT_TRAIN_TEST*len(labeled_wind.loc[
                            labeled_wind['Label'] == 1])), :],
                    labeled_wind.loc[labeled_wind['Label'] == 2].iloc[
                        :round(SPLIT_TRAIN_TEST*len(labeled_wind.loc[
                            labeled_wind['Label'] == 2])), :]]
                training = pd.concat(lst, ignore_index=True)
                lst = [
                    labeled_wind.loc[labeled_wind['Label'] == 0].iloc[
                        round(SPLIT_TRAIN_TEST*len(labeled_wind.loc[
                            labeled_wind['Label'] == 0])):, :],
                    labeled_wind.loc[labeled_wind['Label'] == 1].iloc[
                        round(SPLIT_TRAIN_TEST*len(labeled_wind.loc[
                            labeled_wind['Label'] == 1])):, :],
                    labeled_wind.loc[labeled_wind['Label'] == 2].iloc[
                        round(SPLIT_TRAIN_TEST*len(labeled_wind.loc[
                            labeled_wind['Label'] == 2])):, :]]
                testing = pd.concat(lst, ignore_index=True)
            else:
                training = labeled_wind.iloc[:round(
                         SPLIT_TRAIN_TEST*len(labeled_wind)), :]
                testing = labeled_wind.iloc[round(
                         SPLIT_TRAIN_TEST*len(labeled_wind)):, :]
            x_train = np.array(training.drop(['Label'], axis=1))
            y_train = np.array(training['Label'])
            x_test = np.array(testing.drop(['Label'], axis=1))
            y_test = np.array(testing['Label'])

            # define class weights
            class_weights = class_weight.compute_class_weight(
                          class_weight='balanced', classes=np.unique(y_train),
                          y=y_train)
            class_weights = dict(zip(np.unique(y_train), class_weights))

            # one-hot encodes the labels
            y_train_onehot = tf.keras.utils.to_categorical(y_train, 3)
            y_test_onehot = tf.keras.utils.to_categorical(y_test, 3)

            # train on the training set
            if algorithm == 'Dense':
                model, history, results = model_dense(
                                        x_train, y_train_onehot, x_test,
                                        y_test_onehot, LEARNING_RATE,
                                        BATCH_SIZE, class_weights,
                                        use_class_weights=use_weight)
            elif algorithm == 'Random_Forest':
                model, history = model_random_forest(
                               x_train, y_train, class_weights, n_trees=100,
                               use_class_weights=use_weight)
            elif algorithm == 'Logistic_Regression':
                model, history = model_logistic_regression(
                               x_train, y_train, class_weights,
                               use_class_weights=use_weight)
            elif algorithm == 'LSTM':
                model, history, results = model_lstm(
                                        x_train, y_train_onehot, x_test,
                                        y_test_onehot, LEARNING_RATE,
                                        BATCH_SIZE, class_weights,
                                        use_class_weights=use_weight)
            else:
                sys.exit('No algorithm selected!')

            # test on the test set; predict on the unseen data
            y_train_predicted_prob, y_train_predicted_label, \
                y_test_predicted_prob, y_test_predicted_label, \
                C_predicted_prob, C_predicted_label = predict_using_model(
                                  model, x_train, x_test, unlabeled_wind,
                                  use_predict_proba=USE_PROBA,
                                  reshape_data=USE_RESHAPE, save_results=True)

            # attach latest results to an array
            df_result = pd.concat([pd.DataFrame([[
                      algorithm, use_weight, second, WINDOW_STEP/20000,
                      WINDOW_THRESH*100,
                      Counter(training['Label'])[0],
                      Counter(training['Label'])[1],
                      Counter(training['Label'])[2],
                      Counter(testing['Label'])[0],
                      Counter(testing['Label'])[1],
                      Counter(testing['Label'])[2],
                      Counter(C_predicted_label[1]['Label'])[0],
                      Counter(C_predicted_label[1]['Label'])[1],
                      Counter(C_predicted_label[1]['Label'])[2],
                      Counter(C_predicted_label[2]['Label'])[0],
                      Counter(C_predicted_label[2]['Label'])[1],
                      Counter(C_predicted_label[2]['Label'])[2],
                      Counter(C_predicted_label[3]['Label'])[0],
                      Counter(C_predicted_label[3]['Label'])[1],
                      Counter(C_predicted_label[3]['Label'])[2],
                      get_roc_auc(y_test, y_test_predicted_prob, method='ovr'),
                      get_roc_auc(y_test, y_test_predicted_prob,
                                  method='ovo')]],
                      columns=col), df_result], ignore_index=True)
# save results
df_result.to_csv(PATH + 'df_result', index=False)
