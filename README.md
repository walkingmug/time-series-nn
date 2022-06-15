# Time Series Event Classification with Machine Learning

## Description

Uses Machine Learning for learning and predicting the class of time series events with four classifiers: Fully Connected Neural Network, Random Forest, Logistic Regression and Long Short-Term Memory (LSTM).

## Getting Started

### Dependencies

* Required packages: colorcet (optional), datashader (optional), h5py, keras_tuner, matplotlib, numpy, pandas, sklearn, tensorflow.
* Developed with Python 3.9.12 (anaconda3) in VS Code (Universal).

### Executing program

* Time series files need to be added to the same folder as 'main.py'.
* Run the file 'main.py'

### Changing the data
* The prediction is made on 'C.dat'. You may change which file to predict by setting the filename:
```
df_C = pd.read_csv('C.csv', ',')
```
* The training and validation is made from joining the files 'A.dat' and 'B.dat'. You can change this by setting some other pandas DataFrame:
```
dataset = df_AB
```
The DataFrame uses the columns 'Value' for the current and 'Label' for the 0-1 labels.
* Set the dataset name, which is used for filenames and to name the plots:
```
DATASET_NAME = 'AB'
```
* Set the path to save the files, or leave it as it is to save them in the same folder:
```
PATH = ''
```
* Set the number of features (window size):
```
FEATURES = 20000
```
* Set the number of outputs expected (currently 3: 0: no event, 1: event A, 2: event B):
```
NUM_CLASS = 3
```
* Change the threshold when classifying windows as events:
```
WINDOW_THRESH = 0.8
```

### Making use of other tools

* You can read .mat files using the functions in 'mat_file_loader.py'.
* Data such as 'A1.mat',.. can be plotted using 'plot.py'. The script uses 'mat_file_loader.py' so filenames should be specified there first.
* Some statistics, such as the number of events or avgerage current can be computed from 'stats.py'.
* The classifiers used can be checked in 'classifiers.py'.
* To use automatic hyperparameter tuning use the file 'hyperparameter_tuning.py'.
* Some window tools, such as splitting sequences into windows can be found in 'window_tools.py'.
* To set the path or classifier parameters such as the number of epochs use 'main.py'.

## Help
* The prediction returns labels on indices of windows.
* Keras may require PyDot and GraphViz to be installed:
```
pip install pydot
pip install graphviz
```

## Authors

Vulnet Alija 

## Version History
* 1.0
    * Redesign of algorithms; additional algorithms added
    * Structural improvements following the PEP8 standards
* 0.1
    * Initial Release

## Acknowledgments

* [Classify Images of Clothing](https://www.tensorflow.org/tutorials/keras/classification)
* [Time Series Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)
