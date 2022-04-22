from cmath import nan
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from numpy.lib.stride_tricks import sliding_window_view
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import time
import pickle
from sklearn.utils import class_weight
from itertools import groupby
from pathlib import Path

seconds = 10
window_size = int(seconds*20000)
window_step = int(seconds*2000)
window_threshold = 0.1

num_class = 3
split_train_test = 0.8
batch_size = 32
epochs = 11
learning_rate = 0.0001
print('Window size:',window_size,'| Window step:',window_step,'| Window thresh.:',window_threshold)
col = ['window size','window step', 'window thresh',
                                'Train: no event', 'Train: event A','Train: event B','Test: no event','Test: event A','Test: event B',
                                'Pred C: no event', 'Pred C: event A', 'Pred C: event B', 'AUC: no event','AUC: event A','AUC: event B',]
df_result = pd.DataFrame(columns = col)
# returns a sliding window
def get_window(seq, window_shape = window_size, step = window_step):
    window = sliding_window_view(seq, window_shape)[::step,:]
    return window

# prints basic information
def print_basic_info():
    print('\n')
    # print('Types of input windows (train+test): ', Counter(labeled_wind['Label']))
    print('Types of input windows (train): ', Counter(training['Label']))
    print('Types of input windows (test): ', Counter(testing['Label']))

    print('Types of C windows (predicted): ', Counter(predict_C['Label'])) 
    # print('Training set Accuracy: ', accuracy_score(y_train_predict, y_train) *100,'%') 
    # print('Test set Accuracy: ', accuracy_score(y_test_predict, y_test) *100,'%')
    events_pred = []
    [events_pred.append(i) for i, j in groupby(predict_C['Label'])]
    # print('\nPredicted window #events: ')
    # print('\t#Events 0: ', Counter(events_pred)[0], ' (', round(Counter(events_pred)[0] / sum(Counter(events_pred).values())*100, 2), '% )')
    # print('\t#Events 1: ', Counter(events_pred)[1], ' (', round(Counter(events_pred)[1] / sum(Counter(events_pred).values())*100, 2), '% )')
    # print('\t#Events 2: ', Counter(events_pred)[2], ' (', round(Counter(events_pred)[2] / sum(Counter(events_pred).values())*100, 2), '% )')
    print('\nAUC (test set): ', get_auc(y_test_onehot, y_test_probability))

# plots training accuracy and validation loss
def plot_train_val_loss():
    metric = 'accuracy'
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(path + 'train_val_acc_' + dataset_name + '.png')
    plt.close()
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(path + 'train_val_loss_' + dataset_name + '.png')
    # plt.show()

# plots confusion matrix
def plot_confusion_matrix(y_test, y_test_predict, labels = [0,1,2]):
    cm = confusion_matrix(y_test,y_test_predict,labels=labels)
    figure = sn.heatmap(cm, annot=True, fmt='g')
    plt.title(dataset_name + ' Test set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(path + 'confusion_matrix_' + dataset_name + '.png', dpi=400)
    plt.close()

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    roc_auc = auc(fp, tp)
    plt.plot(100*fp, 100*tp, label=name+' (area = '+str(round(roc_auc,4))+')' , linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = precision_recall_curve(labels, predictions)
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_roc_prc():
    labels = ['no event','event A', 'event B']
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(15,20)
    fig.suptitle('ROC and AUPRC for training and testing labels', fontsize='xx-large')
    # plot roc
    for i in range(num_class):
        axs[i,0].set_title('ROC: ' + labels[i])
        plt.sca(axs[i,0])
        plot_roc("Train", y_train_onehot[:,i], y_train_probability[:,i])
        plot_roc("Test", y_test_onehot[:,i], y_test_probability[:,i], linestyle='--')
        plt.xlim([-5,105])
        plt.ylim([-5,105])
    # plot auprc
    for i in range(num_class):
        axs[i,1].set_title('AUPRC: ' + labels[i])
        plt.sca(axs[i,1])
        plot_prc("Train", y_train_onehot[:,i], y_train_probability[:,i])
        plot_prc("Test", y_test_onehot[:,i], y_test_probability[:,i], linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
    plt.savefig(path+'ROC-AUPRC.png', facecolor='white', transparent=False)
    plt.close()

# returns indices of highly confident predictions for test set
def get_confidence(threshold = 0.7):
    TP, FP, true_confidence, false_confidence1, false_confidence2 = {}, {}, {}, {}, {}
    for i in range(num_class):
        # get indices of True Positive, False Positive
        TP[i] = np.argwhere((np.array(y_test)==i)&(np.array(y_test_predict)==i)).flatten()
        FP[i] = np.argwhere((np.array(y_test)==i)&(np.array(y_test_predict)!=i)).flatten()
        np.intersect1d(TP[i],np.argwhere(y_test_probability[:,i]>=threshold).flatten())
        # filter those indices using probability threshold
        not_class = [j for j in range(num_class) if j!=i]
        true_confidence[i] = np.intersect1d(TP[i],np.argwhere(y_test_probability[:,i]>=threshold).flatten())
        false_confidence1[i] = np.intersect1d(FP[i],np.argwhere(y_test_probability[:,not_class[0]]>=threshold).flatten())
        false_confidence2[i] = np.intersect1d(FP[i],np.argwhere(y_test_probability[:,not_class[1]]>=threshold).flatten())
        # 2: np.argwhere(y_test_probability[FP[2],2]>=threshold).flatten()}
    return true_confidence, false_confidence1, false_confidence2

# plots confident windows denoted by threshold
def plot_confidence(true_confidence, false_confidence1, false_cofidence2):
    # remove previous files
    [f.unlink() for f in Path(path+'/test_set_high_confidence_plot/true_confidence/').glob('*') if f.is_file()]
    [f.unlink() for f in Path(path+'/test_set_high_confidence_plot/false_confidence/').glob('*') if f.is_file()]
    
    [print(i,':',len(true_confidence[i])) for i in true_confidence]
    print('False confidence1:')
    [print(i,':',len(false_confidence1[i])) for i in false_confidence1]
    print('False confidence2:')
    [print(i,':',len(false_confidence2[i])) for i in false_confidence2]

    save_nr = 10
    labels = ['no event', 'event A', 'event B']
    for i in range(num_class):
        len_true = len(true_confidence[i]) if len(true_confidence[i]) <= save_nr else save_nr
        len_false1 = len(false_confidence1[i]) if len(false_confidence1[i]) <= save_nr else save_nr
        len_false2 = len(false_confidence2[i]) if len(false_confidence2[i]) <= save_nr else save_nr
        not_class = [j for j in range(num_class) if j!=i]
        for j in range(len_true):
            plt.title('True confidence: \'' + labels[i] + '\' - Probability: ' + 
                        str(np.round(y_test_probability[true_confidence[i][j]][i],4)))
            plt.plot(x_test[true_confidence[i][j],:], linestyle='-', color='green')
            plt.grid()
            plt.savefig(path+'/test_set_high_confidence_plot/true_confidence/'+labels[i]+'_'+str(j)+'.png')
            plt.close()
        for j in range(len_false1):
            plt.title('False confidence: was \'' + labels[i] + '\', predicted \'' + labels[not_class[0]] + '\' - Probability: ' + 
                        str(np.round(y_test_probability[false_confidence1[i][j]][not_class[0]],4)))
            plt.plot(x_test[false_confidence1[i][j],:], linestyle='-', color='red')
            plt.grid()
            plt.savefig(path+'/test_set_high_confidence_plot/false_confidence/'+labels[i]+'_'+str(j)+'.png')
            plt.close()
        for j in range(len_false2):
            plt.title('False confidence: was \'' + labels[i] + '\', predicted \'' + labels[not_class[1]] + '\' - Probability: ' + 
                        str(np.round(y_test_probability[false_confidence2[i][j]][not_class[1]],4)))
            plt.plot(x_test[false_confidence2[i][j],:], linestyle='-', color='red')
            plt.grid()
            plt.savefig(path+'/test_set_high_confidence_plot/false_confidence/'+labels[i]+'_'+str(j+len_false1)+'.png')
            plt.close()

# returns a dict of AUC in each label 
def get_auc(labels, predictions):
    classes = [0,1,2]
    auc_list = {}
    for i in range(num_class):
        fp, tp, _ = roc_curve(labels[:,i], predictions[:,i])
        roc_auc = auc(fp, tp)
        auc_list[classes[i]] = round(roc_auc ,4)
    return auc_list
    

# get data
df_A = pd.read_csv('A.csv', ',')
df_B = pd.read_csv('B.csv', ',')
df_C = pd.read_csv('C.csv', ',')

df_A.drop(['Time'], inplace=True, axis=1)
df_B.drop(['Time'], inplace=True, axis=1)
df_B['Label'] = df_B['Label'].replace(1, 2)
df_C.drop(['Time'], inplace=True, axis=1)
df_AB = pd.concat([df_A, df_B], ignore_index=True, sort=False)

path = ''
# with open(path + 'df_AB.pkl', 'wb+') as file:
#     pickle.dump(df_AB, file)
# with open(path + 'df_AB.pkl', 'rb') as file:
#     df_AB = pickle.load(file)

# set learning dataset and path
dataset = df_AB
dataset_name = 'AB'

w_sizes = [10,5,2,1,0.5,0.2,0.1]
for size in w_sizes:
    seconds = size
    window_size = int(seconds*20000)
    window_step = int(seconds*2000)
    if size == 10:
        window_threshold = 0.1
    elif size == 5:
        window_threshold = 0.4
    elif size == 2:
        window_threshold = 0.7
    else:
        window_threshold = 0.8

    # split data into windows
    x_window = get_window(dataset['Value'])
    y_window = get_window(dataset['Label'])
    unlabeled_wind = get_window(df_C['Value'])
    unlabeled_wind = pd.DataFrame(unlabeled_wind)
    values_and_labels = []

    # label windows as 0, 1 or 2
    for i in range(len(y_window)):
        if np.count_nonzero(y_window[i]==2) >= (window_threshold * len(y_window[i])):
            values_and_labels.append(list(x_window[i]))
            values_and_labels[-1].append(2)
        elif np.count_nonzero(y_window[i]==1) >= (window_threshold * len(y_window[i])):
            values_and_labels.append(list(x_window[i]))
            values_and_labels[-1].append(1)
        else:
            values_and_labels.append(list(x_window[i]))
            values_and_labels[-1].append(0)
    labeled_wind = pd.DataFrame(values_and_labels)
    labeled_wind.columns = [*labeled_wind.columns[:-1], 'Label']

    # with open(path + 'labeled_wind.pkl', 'wb+') as file:
    #     pickle.dump(labeled_wind, file)
    # with open('/Users/valija/Desktop/Master Thesis/code/WINES-PRED-C/window_method/multi_output/labeled_wind.pkl', 'rb') as file:
    #     labeled_wind = pickle.load(file)

    # take equal parts of each label
    # c = Counter(labeled_wind['Label'])
    # min_key, min_count = min(c.items(), key=itemgetter(1))
    # labeled_wind = pd.concat([
    #                 labeled_wind[labeled_wind['Label'] == 0][:min_count],
    #                 labeled_wind[labeled_wind['Label'] == 1][:min_count],
    #                 labeled_wind[labeled_wind['Label'] == 2][:min_count]])

    # shuffle windows
    labeled_wind = labeled_wind.sample(frac=1, random_state=1)
    labeled_wind.columns = labeled_wind.columns.astype(str)

    # normalize data 0..1
    labeled_wind = (labeled_wind - labeled_wind.min()) / (labeled_wind.max() - labeled_wind.min())
    labeled_wind['Label'] = labeled_wind['Label'] * 2

    # split data into train and validation
    training = labeled_wind.iloc[:round(split_train_test*len(labeled_wind)), :]
    testing = labeled_wind.iloc[round(split_train_test*len(labeled_wind)):, :]
    x_train =  np.array(training.drop(['Label'], axis=1))
    y_train =  np.array(training['Label'])
    x_test =  np.array(testing.drop(['Label'], axis=1))
    y_test =  np.array(testing['Label'])

    # equalize label weight
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    class_weights = dict(zip(np.unique(y_train), class_weights))

    # one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train, 3)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, 3)

    # create NN model
    # model = Sequential()
    # model.add(Flatten())
    # model.add(Dense(50, input_shape=(window_size,), activation='relu'))#, kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    # model.add(Dense(num_class,activation='softmax')) # output layer

    print('----to tf dataset')
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(training, label="Label")
    print('----model')
    model = tfdf.keras.RandomForestModel(num_trees=100)
    # model.compile(loss='categorical_crossentropy', 
                # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics = [tf.keras.metrics.AUC()]) 
    print('----fit')
    history = model.fit(train_ds)
    print('----test_ds')
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(testing, label="Label")
    print('----compile')
    model.compile(metrics=["AUC"])
    print('----evaluate')
    results = model.evaluate(test_ds)

    # save model
    filename = 'model_' + dataset_name + '.sav'
    # joblib.dump(model, path + filename)
    # model = joblib.load(path+filename)   # load saved model

    # predict training data
    y_train_probability = model.predict(x_train)
    print('\nMean train:', y_train_probability.mean(axis=0))
    # pd.DataFrame(y_train_probability).to_csv(path + 'y_train_probability_' + dataset_name + '.csv')     # probability
    y_train_predict = [np.argmax(y_train_probability[i]) for i,_ in enumerate(y_train_probability)]
    # pd.DataFrame(y_train_predict).to_csv(path + 'y_train_predict_' + dataset_name + '.csv')         # labeled

    # predict validation data
    y_test_probability = model.predict(x_test)
    print('Mean test:', y_test_probability.mean(axis=0))
    # pd.DataFrame(y_test_probability).to_csv(path + 'y_test_probability_' + dataset_name + '.csv')       # probability
    y_test_predict = [np.argmax(y_test_probability[i]) for i,_ in enumerate(y_test_probability)]
    # pd.DataFrame(y_test_predict).to_csv(path + 'y_test_predict_' + dataset_name + '.csv')           # labeled

    # predict C windows
    predict_C = model.predict(unlabeled_wind)
    print('Mean C:', predict_C.mean(axis=0))
    predict_C = [np.argmax(predict_C[i]) for i,_ in enumerate(predict_C)]  
    predict_C = pd.DataFrame(predict_C, columns=['Label'])
    # predict_C.to_csv(path + 'predict_C_from_' + dataset_name + '.csv')

    # plot_confusion_matrix(y_test, y_test_predict)
    # print_basic_info()
    true_confidence, false_confidence1, false_confidence2 = get_confidence()
    # plot_confidence(true_confidence, false_confidence1, false_confidence2)


    print('\nAUC (test set): ', get_auc(y_test_onehot, y_test_probability))

    df_result=pd.concat([pd.DataFrame([[seconds,window_step/20000,window_threshold,
                        Counter(training['Label'])[0],Counter(training['Label'])[1],Counter(training['Label'])[2],
                        Counter(testing['Label'])[0],Counter(testing['Label'])[1],Counter(testing['Label'])[2],
                        Counter(predict_C['Label'])[0],Counter(predict_C['Label'])[1],Counter(predict_C['Label'])[2],
                        get_auc(y_test_onehot, y_test_probability)[0],get_auc(y_test_onehot, y_test_probability)[0],
                        get_auc(y_test_onehot, y_test_probability)[0],]],columns=col),df_result],ignore_index=True)

print(df_result)
df_result.to_csv('df_result', index=False)
