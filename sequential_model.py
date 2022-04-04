from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
from numpy.lib.stride_tricks import sliding_window_view
from keras.layers import Dense, Flatten
from keras.models import Sequential
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import joblib


# returns a sliding window
def get_window(seq, window_shape = 20000, step = 2000):
    window = sliding_window_view(seq, window_shape)[::step,:]
    return window

# prints basic information
def print_basic_info():
    print('Assigned labels: ', Counter(labeled_wind['Label']))
    print('Predicted C windows #values: ', Counter(predict_C['Label'])) 
    print('Training set Accuracy: ', accuracy_score(y_train_predict, y_train) *100,"%") 
    print('Validation set Accuracy: ', accuracy_score(y_test_predict, y_test) *100,"%")

# plots training accuracy and validation loss
def plot_train_val_loss():
    metric = "accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(["train", "val"])
    plt.savefig(path + 'train_val_loss_' + dataset_name + '.png')
    plt.close()

# plots confusion matrix
def plot_confusion_matrix(y_test, y_test_predict, labels = [0,1,2]):
    cm = confusion_matrix(y_test,y_test_predict,labels=labels)
    figure = sn.heatmap(cm, annot=True, fmt='g')
    plt.title(dataset_name)
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

# set learning dataset and path
dataset = df_AB
dataset_name = 'AB'
path = ''
features = 20000
num_class = 3
percentage = 0.8

# split data into windows
x_window = get_window(dataset['Value'])
y_window = get_window(dataset['Label'])
unlabeled_wind = get_window(df_C['Value'])
unlabeled_wind = pd.DataFrame(unlabeled_wind)
values_and_labels = []

# label windows as 0, 1 or 2
for i in range(len(y_window)):
    if np.count_nonzero(y_window[i]==2) >= (percentage * len(y_window[i])):
        values_and_labels.append(list(x_window[i]))
        values_and_labels[-1].append(2)
    elif np.count_nonzero(y_window[i]==1) >= (percentage * len(y_window[i])):
        values_and_labels.append(list(x_window[i]))
        values_and_labels[-1].append(1)
    else:
        values_and_labels.append(list(x_window[i]))
        values_and_labels[-1].append(0)
labeled_wind = pd.DataFrame(values_and_labels)
labeled_wind.columns = [*labeled_wind.columns[:-1], 'Label']

# shuffle windows
labeled_wind = labeled_wind.sample(frac=1, random_state=1)
unlabeled_wind = unlabeled_wind.sample(frac=1, random_state=1)

# normalize data 0..1
labeled_wind = (labeled_wind - labeled_wind.min()) / (labeled_wind.max() - labeled_wind.min())
labeled_wind['Label'] = labeled_wind['Label'] * 2
unlabeled_wind = (unlabeled_wind - unlabeled_wind.min()) / (unlabeled_wind.max() - unlabeled_wind.min())

# split data into train and validation
training = labeled_wind.iloc[:round(0.7*len(labeled_wind)), :]
validation = labeled_wind.iloc[round(0.7*len(labeled_wind)):, :]
x_train =  np.array(training.drop(["Label"], axis=1))
y_train =  np.array(training[["Label"]])
x_test =  np.array(validation.drop(["Label"], axis=1))
y_test =  np.array(validation[["Label"]])

# create NN model
model = Sequential()
model.add(Flatten())
model.add(Dense(50,input_shape=(features,),activation='relu'))
model.add(Dense(num_class,activation='softmax')) # output layer
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ['accuracy']) 
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=11,batch_size=32)

# save model
filename = 'model_' + dataset_name + '.sav'
joblib.dump(model, path + filename)
# model = joblib.load(filename)   # load saved model

# predict training data
y_train_predict = model.predict(x_train)
pd.DataFrame(y_train_predict).to_csv(path + "y_train_probability_" + dataset_name + ".csv")     # probability
y_train_predict = [np.argmax(y_train_predict[i]) for i,_ in enumerate(y_train_predict)]
pd.DataFrame(y_train_predict).to_csv(path + "y_train_predict_" + dataset_name + ".csv")         # labeled

# predict validation data
y_test_predict = model.predict(x_test)
pd.DataFrame(y_test_predict).to_csv(path + "y_test_probability_" + dataset_name + ".csv")       # probability
y_test_predict = [np.argmax(y_test_predict[i]) for i,_ in enumerate(y_test_predict)]
pd.DataFrame(y_test_predict).to_csv(path + "y_test_predict_" + dataset_name + ".csv")           # labeled

# predict C windows
print('Predicting C: ', unlabeled_wind.shape)
predict_C = model.predict(unlabeled_wind)
predict_C = [np.argmax(predict_C[i]) for i,_ in enumerate(predict_C)]  
predict_C = pd.DataFrame(predict_C, columns=['Label'])
predict_C.to_csv(path + "predict_C_from_" + dataset_name + ".csv")

plot_train_val_loss()
plot_confusion_matrix(y_test, y_test_predict)
print_basic_info()
