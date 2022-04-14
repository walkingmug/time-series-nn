from cmath import nan
from collections import Counter
from operator import itemgetter
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
from keras.layers import Dense, Flatten
from keras.models import Sequential
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import time
import pickle
from sklearn.utils import class_weight

window_size = 20000
window_step = 2000
num_class = 3
window_threshold = 0.5
split_train_test = 0.8
batch_size = 32 
epochs = 10
learning_rate = 0.0001

# returns a sliding window
def get_window(seq, window_shape = window_size, step = window_step):
    window = sliding_window_view(seq, window_shape)[::step,:]
    return window

# prints basic information
def print_basic_info():
    print('\n')
    print('Assigned window labels: ', Counter(labeled_wind['Label']))
    print('Predicted C window labels: ', Counter(predict_C['Label'])) 
    # print('Training set Accuracy: ', accuracy_score(y_train_predict, y_train) *100,'%') 
    # print('Test set Accuracy: ', accuracy_score(y_test_predict, y_test) *100,'%')
    print('\n')

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

with open(path + 'labeled_wind.pkl', 'wb+') as file:
    pickle.dump(labeled_wind, file)
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

# normalize data 0..1
labeled_wind = (labeled_wind - labeled_wind.min()) / (labeled_wind.max() - labeled_wind.min())
labeled_wind['Label'] = labeled_wind['Label'] * 2

# split data into train and validation
training = labeled_wind.iloc[:round(split_train_test*len(labeled_wind)), :]
validation = labeled_wind.iloc[round(split_train_test*len(labeled_wind)):, :]
x_train =  np.array(training.drop(['Label'], axis=1))
y_train =  np.array(training['Label'])
x_test =  np.array(validation.drop(['Label'], axis=1))
y_test =  np.array(validation[['Label']])

# equalize label weight
class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

# create NN model
model = Sequential()
model.add(Flatten())
model.add(Dense(50, input_shape=(window_size,), activation='relu'))#, kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
# model.add(Dense(25, input_shape=(window_size,), activation='relu'))
model.add(Dense(num_class,activation='softmax')) # output layer
model.compile(loss='sparse_categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics = 'accuracy') 
history = model.fit(x_train,y_train,validation_split=0.2,epochs=epochs,batch_size=batch_size,class_weight=class_weights)
results = model.evaluate(x_test, y_test, batch_size=batch_size)

# save model
filename = 'model_' + dataset_name + '.sav'
joblib.dump(model, path + filename)
# model = joblib.load(path+filename)   # load saved model

# predict training data
y_train_predict = model.predict(x_train)
print('\nMean train:', y_train_predict.mean(axis=0))
pd.DataFrame(y_train_predict).to_csv(path + 'y_train_probability_' + dataset_name + '.csv')     # probability
y_train_predict = [np.argmax(y_train_predict[i]) for i,_ in enumerate(y_train_predict)]
pd.DataFrame(y_train_predict).to_csv(path + 'y_train_predict_' + dataset_name + '.csv')         # labeled

# predict validation data
y_test_predict = model.predict(x_test)
print('Mean test:', y_test_predict.mean(axis=0))
pd.DataFrame(y_test_predict).to_csv(path + 'y_test_probability_' + dataset_name + '.csv')       # probability
y_test_predict = [np.argmax(y_test_predict[i]) for i,_ in enumerate(y_test_predict)]
pd.DataFrame(y_test_predict).to_csv(path + 'y_test_predict_' + dataset_name + '.csv')           # labeled

# predict C windows
predict_C = model.predict(unlabeled_wind)
print('Mean C:', predict_C.mean(axis=0))
predict_C = [np.argmax(predict_C[i]) for i,_ in enumerate(predict_C)]  
predict_C = pd.DataFrame(predict_C, columns=['Label'])
predict_C.to_csv(path + 'predict_C_from_' + dataset_name + '.csv')

plot_confusion_matrix(y_test, y_test_predict)
print_basic_info()
plot_train_val_loss()
