import numpy as np
import pandas as pd
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean

_SQRT2 = np.sqrt(2)
import math
import cv2
# from emd_computation import emd_calc
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy import stats



data1 = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Chart-Analyzer Scatt/Data/Synthetic/sc15/sc15.csv",  sep=",", index_col=False).to_numpy()
data2 = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Chart-Analyzer Scatt/Data/Synthetic/sc15/data_sc15.csv",  sep=",", index_col=False).to_numpy()

dx = max(data1[:,0])-min(data1[:,0])
dy = max(data1[:,1])-min(data1[:,1])

# TPc = 0
# FPc = 0
# ids = list(range(len(data2)))
# for i in range(len(data1)):
#     flag = True
#     for j in ids:
#         if (abs(data1[i][0]-data2[j][0])/dx <= 0.02) and (abs(data1[i][1]-data2[j][1])/dy <= 0.02):
#             flag = False
#             ids.remove(j)
#             TPc += 1
#             break
#     if flag:
#         FPc +=1
# FNc= len(ids)

TPc = 0
FNc = 0
ids = list(range(len(data1)))
for i in range(len(data2)):
    flag = True
    for j in range(len(data1)):
        if (abs(data2[i][0]-data1[j][0])/dx <= 0.02) and (abs(data2[i][1]-data1[j][1])/dy <= 0.02):
            flag = False
            if j in ids:
                ids.remove(j)
            TPc += 1
            break
    if flag:
        FNc +=1
FPc= len(ids)

print(len(data1),len(data2),TPc,FPc,FNc)

prec = TPc/(TPc+FPc)
recall = TPc/(TPc+FNc)
F1src = 2*prec*recall/(prec+recall)


print(prec,recall,F1src)
