from matplotlib.pyplot import get
from file import *
from stats import *
from plot import *
from tabulate import tabulate
import time
import csv

starttime = time.time()

# pred = get_predictions(data_B)
# print('Getting indices...')
# ind = get_index_sign(data_B)

# for data_subgroup in data_B:
#     print('Prediction: ', data_subgroup, '= ', len(pred[data_subgroup]))
#     print('Actual: ', data_subgroup, '= ', len(ind[data_subgroup]))

# f = open( 'prediction-var.py', 'w+' )
# f.write( 'dict = ' + repr(pred) + '\n' )
# f.close()


# getplot_prediction(data_A, 'data_A')
# for i in data_B:
#     data = get_file(i)
#     print(i, '=', len(get_slopes(data[1])))


# # get indices of events
# dataind = {}
# for subgroup in data_A:
#     data = get_file(subgroup) 
#     dataind[subgroup] = data[1]  

# ind = get_index_sign(data_A)
# writer = csv.writer(open("A-ones.csv", 'w+'))
# for group in ind:
#     for i in range(len(ind[group])):
#         towrite = dataind[group][ind[group][i][0]:ind[group][i][1]]
#         towrite = np.append(towrite,1)

#         writer.writerow(towrite)


import pandas as pd

df = pd.read_csv('../A-dat.csv')
# df.head(1000000).to_csv("A1.csv")
df.iloc[:1000000,:].to_csv('A1.csv', index=False)

# countA = get_count(data_A)
# countB = get_count(data_B)
# groupsA = get_groups(data_A)
# groupsB = get_groups(data_B)
# durA = get_duration(data_A)
# durB = get_duration(data_B)
# currentA = get_current(data_A)
# currentB = get_current(data_B)
# currentA_all = get_current_all(data_A)
# currentB_all = get_current_all(data_B)
# currentC_all = get_current_all(data_C)

# mydata = []
# for i in range(len(data_A)):
#     mydata.append([data_A[i],countA[data_A[i]][0],countA[data_A[i]][1],groupsA[data_A[i]],\
#         round(mean(durA[data_A[i]]),4), round(mean(currentA_all[data_A[i]]),4)])
# for i in range(len(data_B)):
#     mydata.append([data_B[i],countB[data_B[i]][0],countB[data_B[i]][1],groupsB[data_B[i]],\
#         round(mean(durB[data_B[i]]),4), round(mean(currentB_all[data_B[i]]),4)])
# for i in range(len(data_C)):
#     mydata.append([data_C[i],'-','-','-', '-', mean(currentC_all[data_C[i]])])
# head = ['group','total #0','total #1','# of events','mean event duration (s)', 'mean current (pA)']

# print(tabulate(mydata, headers=head, tablefmt="grid"))

# with open('statistics.txt', 'w') as f:
#     f.write(tabulate(mydata))

# writer = csv.writer(open("statistics.csv", 'w+'))
# writer.writerow(head)
# for row in mydata:
#     writer.writerow(row)

endtime = time.time()
print('Time: ',round(endtime-starttime,0), 's')