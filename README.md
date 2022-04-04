# Time Series Classification

## Description

Uses Deep Neural Networks for learning and predicting the class of time series.

## Getting Started

### Dependencies

* Requires TensorFlow, Keras, Scikit-Learn, Pandas, Seaborn, Matplotlib, Joblib.
* Developed with Python 3.10.2 in VS Code 1.65.2 (Universal).
* Developed on macOS Catalina 10.15.7.

### Executing program

* Files 'A.dat', 'B.dat' and 'C.dat' need to be added to the same folder as 'sequential_model.py'
* Run the file 'sequential_model.py'

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
dataset_name = 'AB'
```
* Set the path to save the files, or leave it as it is to save them in the same folder:
```
path = ''
```
* Set the number of features (window size):
```
features = 20000
```
* Set the number of outputs expected (currently 3: 0: no event, 1: event A, 2: event B):
```
num_class = 3
```
* Change the threshold when classifying windows as events:
```
percentage = 0.8
```

### Making use of other tools

* You can read .mat files using the functions in 'file.py'.
* Data such as 'A1.mat',.. can be plotted using 'plot.py'. The script uses 'file.py' so filenames should be specified there first.
* Some statistics, such as #events or avgerage current can be computed from 'stats.py'.
* The above scripts may be executed from 'main.py'.

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

* 0.1
    * Initial Release

## Acknowledgments

* [Classify Images of Clothing](https://www.tensorflow.org/tutorials/keras/classification)
* [Time Series Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)
