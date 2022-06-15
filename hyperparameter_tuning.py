"""Automatic hyperparameter tuning in Sequential models using KerasTuner.
KerasTuner available at: https://keras.io/keras_tuner/
"""
import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf

import main


def model_builder(hp):
    """Uses the hyperparameters defined inside to hypertune the model.

    Args:
        hp (_type_): _description_

    Returns:
        History callback: The compiled model.
    """
    model = keras.Sequential()

    # Choose the range of units to be used in the first layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(
        units=hp_units, activation='relu', input_shape=(40000, )))
    model.add(keras.layers.Dense(main.NUM_CLASS, activation='softmax'))

    # Choose the range of learning rates for the optimizer
    hp_learning_rate = hp.Choice(
                     'learning_rate', values=[1e-0, 1e-1, 1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.AUC()])

    return model


def model_dense_tuner(
        x_train, y_train_onehot, x_test, y_test_onehot, class_weight_list):
    """Trains a Fully Connected Neural Network model.

    Args:
        x_train (numpy.ndarray): Values of the train set.
        y_train_onehot (numpy.ndarray): Labels of the train set (one-hot).
        x_test (numpy.ndarray): Values of the test set.
        y_test_onehot (numpy.ndarray): Labels of the test set (one-hot).
        class_weights (dict): Balances the weights.

    Returns:
        History callback: The trained model.
    """
    stop_early = tf.keras.callbacks.EarlyStopping(
               monitor='val_loss', patience=5)

    tuner = kt.Hyperband(
          model_builder,
          objective=kt.Objective("auc", direction="max"),
          max_epochs=10,
          factor=3,
          directory='tunings',
          project_name='best_params')

    tuner.search(x_train, y_train_onehot, epochs=50, validation_split=0.2,
                 callbacks=[stop_early], class_weight=class_weight_list)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""The hyperparameter search is complete. The optimal number of units
        in the first densely-connected layer is {best_hps.get('units')} and
        the optimal learning rate for the optimizer is
        {best_hps.get('learning_rate')}.""")

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
            x_train, y_train_onehot, epochs=50, validation_split=0.2,
            class_weight=class_weight_list)

    val_acc_per_epoch = history.history['val_auc_1']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch, ))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(
        x_train, y_train_onehot, epochs=best_epoch, validation_split=0.2,
        class_weight=class_weight_list)

    eval_result = hypermodel.evaluate(x_test, y_test_onehot)
    print("[test loss, test accuracy]:", eval_result)
    model = hypermodel

    return model
