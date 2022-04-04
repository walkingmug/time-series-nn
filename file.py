import h5py


# get the file
def get_file(filename):
    f = h5py.File('../data/FRI_sledi/' + filename + '.mat')
    data = f[filename]
    return data


# put all filenames here
data_A = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A8', 'A9', 'A10']
data_B = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']
data_C = ['C1', 'C2', 'C3', 'C4', 'C6']