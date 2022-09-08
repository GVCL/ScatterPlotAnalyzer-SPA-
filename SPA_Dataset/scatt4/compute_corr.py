import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

dataO = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Chart-Analyzer Scatt/Data/Synthetic/m3/m3.csv",  sep=",", index_col=False)
dataR = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Chart-Analyzer Scatt/Data/Synthetic/m3/data_m3.csv",  sep=",", index_col=False)

print(dataO.corr(),dataR.corr())


# df = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Chart-Analyzer Scatt/Data/Synthetic/m1.csv")
# xlabels = (df.loc[ : , list(df)[0]]).values
# ylabels = list(df)[1:len(list(df))]
# data = (df.loc[ : , ylabels]).values
# x = np.arange(len(xlabels))  # the label locations
#
# fig, ax = plt.subplots()
# for i in range(len(ylabels)):
#     ax.scatter(x, data[:,i],  label=ylabels[i])
# # Add some text for labels, title and custom x-axis tick labels, etc.
# plt.xlabel('Scores')
# # plt.title(df['Title'][0])
# # plt.xticks(x, xlabels, fontsize=8)
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# fig.tight_layout()
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Chart-Analyzer Scatt/Data/Synthetic/m1.png')
#
# plt.show()
