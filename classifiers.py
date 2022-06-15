"""Define the classifiers and the prediction function.
"""
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import sklearn
import tensorflow as tf
import pandas as pd
import numpy as np

import main


def model_dense(
        x_train, y_train_onehot, x_test, y_test_onehot, learning_rate,
        batch_size, class_weights, use_class_weights=False):
    """Trains a Fully Connected Neural Network model.

    Args:
        x_train (numpy.ndarray): Values of the train set.
        y_train_onehot (numpy.ndarray): Labels of the train set (one-hot).
        x_test (numpy.ndarray): Values of the test set.
        y_test_onehot (numpy.ndarray): Labels of the test set (one-hot).
        learning_rate (float): Value of the learning rate.
        batch_size (int): Value of the batch size.
        class_weights (dict): Balances the weights.
        use_class_weights (bool, optional): Whether or not to use the weights.
        Defaults to False.

    Returns:
        History callback: The trained model.
        History callback: The history of the trained model.
        History callback: The history of the evaluation of the model.
    """
    # create a model with one input, one hidden (FCNN) and one output layer
    model = Sequential()
    model.add(Dense(50, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Dense(main.NUM_CLASS, activation='softmax'))

    # configure parameters
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=learning_rate),
                  metrics=[tf.keras.metrics.AUC()])

    # train the model on the train set
    if use_class_weights is True:
        history = model.fit(
                x_train, y_train_onehot, validation_split=0.2,
                epochs=main.EPOCHS, batch_size=batch_size,
                class_weight=class_weights)
    else:
        history = model.fit(
                x_train, y_train_onehot, validation_split=0.2,
                epochs=main.EPOCHS, batch_size=batch_size)

    # evaluate the model on the test set
    results = model.evaluate(x_test, y_test_onehot, batch_size=batch_size)

    return model, history, results


def model_random_forest(
        x_train, y_train, class_weights, n_trees=100, use_class_weights=False):
    """Trains a Random Forest model.

    Args:
        x_train (numpy.ndarray): Values of the train set.
        y_train (numpy.ndarray): Labels of the train set.
        class_weights (dict): Balances the weights.
        n_trees (int, optional): Number of trees. Defaults to 100.
        use_class_weights (bool, optional): Whether or not to use the weights.
        Defaults to False.

    Returns:
        History callback: The trained model.
        History callback: The history of the trained model.
    """
    # create the model
    if use_class_weights is True:
        model = sklearn.ensemble.RandomForestClassifier(
              n_estimators=n_trees, class_weight=class_weights)
    else:
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_trees)

    # train the model on the train set
    history = model.fit(x_train, y_train)

    return model, history


def model_logistic_regression(
        x_train, y_train, class_weights, use_class_weights=False):
    """Trains a Logistic Regression model.

    Args:
        x_train (numpy.ndarray): Values of the train set.
        y_train (numpy.ndarray_): Labels of the train set.
        class_weights (_type_): _Balances the weights.
        use_class_weights (bool, optional): Whether or not to use the weights.
        Defaults to False.

    Returns:
                History callback: The trained model.
        History callback: The history of the trained model.
    """
    # create the model
    if use_class_weights is True:
        model = sklearn.linear_model.LogisticRegression(
              class_weight=class_weights)
    else:
        model = sklearn.linear_model.LogisticRegression()

    # train the model on the train set
    history = model.fit(x_train, y_train)

    return model, history


def model_lstm(
        x_train, y_train_onehot, x_test, y_test_onehot, learning_rate,
        batch_size, class_weights, use_class_weights=False):
    """Trains a Long Short-Term Memory model.

    Args:
        x_train (numpy.ndarray): Values of the train set.
        y_train_onehot (numpy.ndarray): Labels of the train set (one-hot).
        x_test (numpy.ndarray): Values of the test set.
        y_test_onehot (numpy.ndarray): Labels of the test set (one-hot).
        learning_rate (float): Value of the learning rate.
        batch_size (int): Value of the batch size.
        class_weights (dict): Balances the weights.
        use_class_weights (bool, optional): Whether or not to use the weights.
        Defaults to False.

    Returns:
        History callback: The trained model.
        History callback: The history of the trained model.
        History callback: The history of the evaluation of the model.
    """
    # reshape the data from 2D to 3D
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    # create a model with one input, one hidden (LSTM) and one output layer
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(x_train.shape[1], x_train.shape[2]),
                   activation='relu'))
    model.add(Dense(main.NUM_CLASS, activation='softmax'))

    # configure parameters
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.AUC()])

    # train the model on the train set
    if use_class_weights is True:
        history = model.fit(
                x_train, y_train_onehot, validation_split=0.2,
                epochs=main.EPOCHS, batch_size=batch_size,
                class_weight=class_weights)
    else:
        history = model.fit(
                x_train, y_train_onehot, validation_split=0.2,
                epochs=main.EPOCHS, batch_size=batch_size)

    # evaluate the model on the test set
    results = model.evaluate(x_test, y_test_onehot, batch_size=batch_size)

    return model, history, results


def predict_using_model(
        model, x_train, x_test, unlabeled_wind, save_results=True,
        use_predict_proba=False, reshape_data=False):
    """Uses the trained model to make predictions on the unseen data.

    Args:
        model (History callback): A trained model.
        x_train (numpy.ndarray): Values of the train set.
        x_test (numpy.ndarray): Labels of the train set.
        unlabeled_wind (numpy.ndarray):Values of the unseen set.
        save_results (bool, optional): Choose whether to save preditions.
        Defaults to True.
        use_predict_proba (bool, optional): Some models require different
        prediction functions. Defaults to False.
        reshape_data (bool, optional): Converts 2D data to 3D (for LSTM).
        Defaults to False.

    Returns:
        numpy.ndarray: Probabilities of the predictions on training set.
        numpy.ndarray: Labels of the predictions on training set.
        numpy.ndarray: Probabilities of the predictions on test set.
        numpy.ndarray: Labels of the predictions on test set.
        dict: Probabilities of the predictions on the unseen set.
        dict: Labels of the predictions on the unseen set.
    """
    # reshape the data to 3D if LSTM is used
    if reshape_data is True:
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        unlabeled_wind_reshape = []
        for i, dframe in enumerate(unlabeled_wind):
            unlabeled_wind_reshape.append(
                np.array(dframe).reshape(dframe.shape[0], 1, dframe.shape[1]))
        unlabeled_wind = unlabeled_wind_reshape

    # predict training set
    if use_predict_proba is False:
        y_train_predicted_prob = model.predict(x_train)
    else:
        model.predict_proba(x_train)
    y_train_predicted_label = [
                            np.argmax(y_train_predicted_prob[i]) for i, _ in
                            enumerate(y_train_predicted_prob)]

    # predict test set
    if use_predict_proba is False:
        y_test_predicted_prob = model.predict(x_test)
    else:
        model.predict_proba(x_test)
    y_test_predicted_label = [
                           np.argmax(y_test_predicted_prob[i]) for i, _ in
                           enumerate(y_test_predicted_prob)]

    # predict unseen data. Returs dict {key = C type : value = value}
    C_predicted_prob = {}
    C_predicted_label = {}
    for i, dframe in enumerate(unlabeled_wind):
        if use_predict_proba is False:
            C_predict_prob = model.predict(dframe)
        else:
            model.predict_proba(dframe)
        C_predict_label = [
                        np.argmax(C_predict_prob[i]) for i, _ in
                        enumerate(C_predict_prob)]
        C_predict_label = pd.DataFrame(
                        np.array(C_predict_label), columns=['Label'])
        C_predicted_prob[i+1] = C_predict_prob
        C_predicted_label[i+1] = C_predict_label

    # save the predicitons of the windows
    if save_results is True:
        pd.DataFrame(np.array(y_train_predicted_prob)).to_csv(
            main.PATH + 'y_train_probability_' + main.DATASET_NAME + '.csv')
        pd.DataFrame(np.array(y_train_predicted_label)).to_csv(
            main.PATH + 'y_train_predict_' + main.DATASET_NAME + '.csv')
        pd.DataFrame(np.array(y_test_predicted_prob)).to_csv(
            main.PATH + 'y_test_predicted_prob_' + main.DATASET_NAME + '.csv')
        pd.DataFrame(np.array(y_test_predicted_label)).to_csv(
            main.PATH + 'y_test_predict_' + main.DATASET_NAME + '.csv')
        for _, key in enumerate(C_predicted_prob):
            pd.DataFrame(np.array(C_predicted_prob[key])).to_csv(
                main.PATH + 'C_predicted_prob_C' + str(key) +
                main.DATASET_NAME + '.csv')
        for _, key in enumerate(C_predicted_label):
            pd.DataFrame(np.array(C_predicted_label[key])).to_csv(
                main.PATH + 'C_predicted_label_C' + str(key) +
                main.DATASET_NAME + '.csv')

    return y_train_predicted_prob, y_train_predicted_label, \
        y_test_predicted_prob, y_test_predicted_label, C_predicted_prob, \
        C_predicted_label
