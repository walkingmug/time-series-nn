from file import *
from stats import *
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import datashader as ds
import pandas as pd
import colorcet as cc



# plot the data considering events
def getplot_separate(data_group, group_name):
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        sample_freq = 20000
        type_one_index = np.where(data[2] == 1)[0]
        #type_one_index = [x+1 for x in type_one_index]   # starts counting at 0, so we increase by one to match the data counting
        type_one_value = np.copy(data[1][type_one_index, ])
        type_zero_index = np.where(data[2] == 0)[0]
        type_zero_value = np.copy(data[1][type_zero_index, ])
        type_one_index = [x/sample_freq for x in type_one_index]
        # type_zero_index = [x/sample_freq for x in data[0]]
        type_zero_index = [x/sample_freq for x in type_zero_index]

        # plt.plot(type_zero_index, data[1], color = 'b', linestyle = 'none', marker = '.', label = '0')
        # plt.plot(type_one_index, type_one_value, color = 'r', linestyle = 'none', marker = '.', label = '1')
        # plt.scatter(type_zero_index, data[1], color = 'b', s=1, label = '0')
        plt.scatter(type_zero_index, type_zero_value, color = 'b', s=1, label = '0')
        plt.scatter(type_one_index, type_one_value, color = 'r', s=1, label = '1')
        plt.ticklabel_format(style='plain')
        plt.ylabel('current [pA]')
        plt.xlabel('time [s]')
        plt.grid()
        plt.title(data_subgroup)
        plt.legend(loc='lower right')
        plt.savefig(group_name + '/' + data_subgroup + '_separate.png', facecolor='white', transparent=False)
        plt.close()


# plot the data considering events, in datashader
def getplot_shade(data_group, group_name):
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        sample_freq = 20000
        type_one_index = np.where(data[2] == 1)[0]
        type_one_value = np.copy(data[1][type_one_index, ])
        type_zero_index = np.where(data[2] == 0)[0]
        type_zero_value = np.copy(data[1][type_zero_index, ])
        type_one_index = [x/sample_freq for x in type_one_index]
        type_zero_index = [x/sample_freq for x in type_zero_index]

        zeros_new = np.stack((type_zero_index, type_zero_value), axis=1)
        ones_new = np.stack((type_one_index, type_one_value), axis=1)

        # ones_new = np.stack((data[0],data[1]), axis = 1)

        df = pd.DataFrame(data=ones_new, columns=["x", "y"])  # create a DF from array
        cvs = ds.Canvas(plot_width=500, plot_height=500)  # auto range or provide the `bounds` argument
        agg = cvs.points(df, 'x', 'y')  # this is the histogram
        img = ds.tf.set_background(ds.tf.shade(agg, how="log", cmap=cc.fire), "black").to_pil()  # create a rasterized image
    
        plt.imshow(img)
        plt.savefig(group_name + '/' + data_subgroup + '_shade.png', facecolor='white', transparent=False)
        plt.close()


# plot the data not considering events
def getplot(data_group, group_name):
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        plt.plot(data[0], data[1], color = 'b', linestyle = 'none', marker = '.', markersize=1)
        plt.ticklabel_format(style='plain')
        plt.title(data_subgroup)
        plt.ylabel('current [pA]')
        plt.savefig(group_name + '/' + data_subgroup + '.png', facecolor='white', transparent=False)
        plt.close()


# plot histogram of currents
def getplot_bar(data_group, group_name):
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        
        plt.hist(data[1], histtype='bar', ec='black')
        plt.title(data_subgroup)
        plt.xlabel('current [pA]')
        plt.savefig(group_name + '/' + data_subgroup + '_bar' + '.png', facecolor='white', transparent=False)
        plt.close()

# plot only events, with some non-events on the left and right
def getplot_spec(data_group, group_name):
    duration = get_duration(data_group)
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        x=data[2]
        x[x==0.0] = -1  
        zero_crossings = np.where(np.diff(np.signbit(x)))[0]    # check where value changes from postive to negative, and vice versa
        sample_freq = 20000
        y = [x/sample_freq for x in data[0]]

        for i in range(0,len(zero_crossings),2):
            padding = math.floor((zero_crossings[i+1]-zero_crossings[i])/7)     # change value to higher for narrower plot

            # plt.scatter(y[zero_crossings[i]-padding:zero_crossings[i]],\
            #     data[1][zero_crossings[i]-padding:zero_crossings[i]],\
            #     color = 'b', s=1)
            # plt.scatter(y[zero_crossings[i+1]:zero_crossings[i+1]+padding],\
            #     data[1][zero_crossings[i+1]:zero_crossings[i+1]+padding],\
            #     color = 'b', s=1, label = '0')
            # plt.scatter(y[zero_crossings[i]:zero_crossings[i]+padding], data[1][zero_crossings[i]:zero_crossings[i]+padding],\
            #     color = 'r', s=1)      
            # plt.scatter(y[zero_crossings[i+1]-padding:zero_crossings[i+1]], data[1][zero_crossings[i+1]-padding:zero_crossings[i+1]],\
            #     color = 'r', s=1, label = '1')  

            plt.scatter(y[zero_crossings[i]-padding:zero_crossings[i]],\
                data[1][zero_crossings[i]-padding:zero_crossings[i]],\
                color = 'b', s=1)
            plt.scatter(y[zero_crossings[i+1]:zero_crossings[i+1]+padding],\
                data[1][zero_crossings[i+1]:zero_crossings[i+1]+padding],\
                color = 'b', s=1, label = '0')
            plt.scatter(y[zero_crossings[i]:zero_crossings[i+1]], data[1][zero_crossings[i]:zero_crossings[i+1]],\
                color = 'r', s=1, label = '1')            
            plt.ticklabel_format(style='plain')
            plt.ylabel('current [pA]')
            plt.xlabel('time [s]')
            plt.grid()
            plt.title(data_subgroup + '-' + str(math.ceil(i/2)) + ': ' + str(duration[data_subgroup][math.ceil(i/2)]) + 's')
            plt.legend(loc = 'lower right')
            Path(group_name + '/' + data_subgroup + '_spec/').mkdir(parents=True, exist_ok=True)
            plt.savefig(group_name + '/' + data_subgroup + '_spec/' + data_subgroup + '-' + str(math.ceil(i/2)) + '_spec.png', facecolor='white', transparent=False)
            plt.close()
            print("done " + data_subgroup)


# plot the data considering events, in datashader
def getplot_spec_shade(data_group, group_name):
    for data_subgroup in data_group:
        data = get_file(data_subgroup)

        x=data[2]
        x[x==0.0] = -1  
        zero_crossings = np.where(np.diff(np.signbit(x)))[0]    # check where value changes from postive to negative, and vice versa
        sample_freq = 20000
        datasec = [x/sample_freq for x in data[0]]

        for i in range(0,len(zero_crossings),2):
            ones_index = datasec[zero_crossings[i]:zero_crossings[i+1]]
            ones_value = data[1][zero_crossings[i]:zero_crossings[i+1]]
            ones_new = np.stack((ones_index, ones_value), axis=1)

            df = pd.DataFrame(data=ones_new, columns=["x", "y"])  # create a DF from array
            cvs = ds.Canvas(plot_width=500, plot_height=500)  # auto range or provide the `bounds` argument
            agg = cvs.points(df, 'x', 'y')  # this is the histogram
            img = ds.tf.set_background(ds.tf.shade(agg, how="log", cmap=cc.fire), "black").to_pil()  # create a rasterized image
        
            plt.imshow(img)
            plt.savefig(group_name + '/' + data_subgroup + '_spec/' + data_subgroup + '-' + str(math.ceil(i/2)) + '_shade.png', facecolor='white', transparent=False)
            plt.close()


# plots the predictions
def getplot_prediction(data_group, group_name):
    for data_subgroup in data_group:
        data = get_file(data_subgroup)
        slopes = get_slopes(data[1])

        plt.scatter(data[0], data[1], color = 'b', s=1, label = '0')
        for i in range(len(slopes)):
            plt.scatter(data[0][slopes[i][0]:slopes[i][1]], data[1][slopes[i][0]:slopes[i][1]], color = 'r', s=1)
        plt.ticklabel_format(style='plain')
        plt.ylabel('current [pA]')
        plt.xlabel('time [s]')
        plt.grid()
        plt.title(data_subgroup)
        plt.legend(loc='lower right')
        plt.savefig(group_name + '/' + data_subgroup + '_pred.png', facecolor='white', transparent=False)
        plt.close()
        print(group_name + '/' + data_subgroup + '_pred.png')