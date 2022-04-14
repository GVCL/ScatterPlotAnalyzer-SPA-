import xml.etree.ElementTree as ET
from Retrieve_Text import get_text_labels,get_title,get_xtitle,get_ytitle,get_legends
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import *
from operator import add
import seaborn as sns
import csv
import math
import cv2
import os
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def scatter(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename) + '/'
    img = cv2.imread(path+image_name+".png")
    root = ET.parse(path+image_name+'.xml').getroot()
    data_tensors = pd.read_csv(path + "tensor_vote_matrix_" + image_name + ".csv", sep=",", index_col=False)
    data_clr = pd.read_csv(path + "Image_RGB_" + image_name + ".csv", sep=",", index_col=False)

    # Now group stack heights based on it's catogery
    # group_colors, group_leg_labels = get_legends(img, root)
    group_colors = [[0,0,0]]
    group_leg_labels = ['']
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())
    cord_list = {
            "x_val": [],
            "y_val": [],
            "CL": []
        }
    a = list(map(add, data_tensors["val1"], data_tensors["val2"]))
    amin, amax = min(a), max(a)
    for i, val in enumerate(a):
        a[i] = (val - amin) / (amax - amin)

    for i in range(X * Y):
        if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.4) and (a[i] > 0.003):
            if (data_tensors["X"][i] < (X - 5) and data_tensors["X"][i] > 5 and data_tensors["Y"][i] > 5 and data_tensors["Y"][i] < (Y - 5)):
                cord_list["x_val"].append(data_tensors["X"][i])
                cord_list["y_val"].append(data_tensors["Y"][i])
                cord_list["CL"].append((data_tensors["CL"][i]))

    cord_list['color'] = []
    for p in range(len(cord_list['x_val'])):
        cord_list['color'].append(data_clr.loc[(data_clr['X'] == cord_list['x_val'][p]) &
                                               (data_clr['Y'] == cord_list['y_val'][p])][['Red', 'Green', 'Blue']].values)


    data = np.array([cord_list['x_val'], cord_list['y_val']]).T

    # db = DBSCAN(eps=3, min_samples=2).fit(data) # scatt5
    # db = DBSCAN(eps=9, min_samples=4).fit(data)  # scatt7
    db = DBSCAN(eps=5, min_samples=4).fit(data) # scatt
    # db = DBSCAN(eps=8, min_samples=4).fit(data) # scatt

    labels = db.labels_
    centers = []
    for i in (np.unique(labels)):
        if i<0:
            continue
        indexes = [id for id in range(len(labels)) if labels[id] == i]
        x = 0
        y = 0
        for k in indexes:
            x += cord_list['x_val'][k]
            y += cord_list['y_val'][k]
        centers += [[x // len(indexes), (y // len(indexes))]]

    img1 = img[::-1, :, :]
    group_id = []
    remove_ids = []

    # plot data with seaborn
    fig, ax = plt.subplots()
    plt.scatter(cord_list["x_val"],  cord_list["y_val"], s=10, c='b')
    plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], s=10, c='r', alpha=0.7)
    plt.axis('off')
    plt.show()

    group_id=[0 for i in range(len(centers))]
    centers = np.delete(np.array(centers),remove_ids,axis=0)

    ''' Map pixels to original coordinates'''
    Xlabel,Ylabel,xbox_centers,ybox_centers = get_text_labels(img,root)
    if(isinstance(Ylabel[0], str) and Ylabel[0].isnumeric()):
        Ylabel = [int(i) for i in Ylabel if i.isnumeric()]
        for i in np.unique(Ylabel):
            id=[j for j, val in enumerate(Ylabel) if i==val]
            if(len(id)==2):
                if(ybox_centers[id[0]][1]<ybox_centers[id[1]][1]):
                    Ylabel[id[1]]*=-1
                    neg_ids=np.where(ybox_centers[:,1] > ybox_centers[id[1]][1])[0]
                else:
                    Ylabel[id[0]]*=-1
                    neg_ids=np.where(ybox_centers[:,1] > ybox_centers[id[0]][1])[0]
                for i in neg_ids:
                    Ylabel[i]*=-1

    if(isinstance(Xlabel[0], str) and Xlabel[0].isnumeric()):
        Xlabel = [int(i) for i in Xlabel if i.isnumeric()]
    xbox_center =xbox_centers[:,0]
    for i in np.unique(Xlabel):
        id=[j for j, val in enumerate(Xlabel) if i==val]
        if(len(id)==2):
            if(xbox_center[id[0]]>xbox_center[id[1]]):
                Xlabel[id[1]]*=-1
                neg_ids=np.where(xbox_center < xbox_center[id[1]])[0]
            else:
                Xlabel[id[0]]*=-1
                neg_ids=np.where(xbox_center < xbox_center[id[0]])[0]
            for i in neg_ids:
                Xlabel[i]*=-1
    print(ybox_centers,Ylabel,Xlabel,xbox_centers)
    print(centers)
    '''normalize center to obtained labels'''
    centers = centers.astype(float)
    xbox_centers, Xlabel = (list(xbox_centers),list(Xlabel))
    normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0][0]-xbox_centers[1][0])
    centers[:,0] = ((centers[:,0] - xbox_centers[0][0]) * normalize_scalex) + Xlabel[0]
    # print("\n----------",centers,"\n----------")
    t = np.array(sorted(np.concatenate((ybox_centers, np.array([Ylabel]).T), axis=1), key=lambda x: x[1]))#, reverse= True))
    ybox_centers,Ylabel = (t[:,0:2],list(t[:,2]))
    normalize_scaley = (Ylabel[0]-Ylabel[1])/(ybox_centers[0][1]-ybox_centers[1][1])
    centers[:, 1] = ((Y - centers[:, 1] - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]
    # print(centers)



    centers_x = []
    centers_y = []
    centers_clr=[]
    centers_labels=[]
    for i in centers:
        centers_x.append(i[0])
        centers_y.append(i[1])
    for i in range(len(group_id)):
        centers_clr.append(group_colors[group_id[i]])
        centers_labels.append(group_leg_labels[group_id[i]])

    '''Reconstruct Chart'''
    colors=ListedColormap(group_colors)
    plt.scatter(centers_x, centers_y, c=group_id, cmap=colors)
    plt.xlabel(get_xtitle(img, root))
    plt.ylabel(get_ytitle(img, root))
    plt.title(get_title(img, root))
    plt.tight_layout()
    plt.savefig(path + "reconstructed_" + image_name + ".png")
    plt.show()

    centers_x, centers_y, group_id = zip( *sorted(zip(centers_x, centers_y, group_id)) )

    # Writing data to CSV file
    L=[]
    for j in np.unique(group_id):
        L = L + [['X']+[group_leg_labels[j]]]
        L = L + [[centers_x[i], centers_y[i]] for i in range(len(centers_x)) if group_id[i] == j]

    with open(path+'data_'+image_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)

    print("Chart Reconstruction Done")

def scatter_multi(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename) + '/'
    img = cv2.imread(path+image_name+".png")
    root = ET.parse(path+image_name+'.xml').getroot()
    data_tensors = pd.read_csv(path + "tensor_vote_matrix_" + image_name + ".csv", sep=",", index_col=False)
    data_clr = pd.read_csv(path + "Image_RGB_" + image_name + ".csv", sep=",", index_col=False)

    # data_tensors = pd.read_csv("/home/komaldadhich/Desktop/test/multiclass_scatter/scm7/lab/tensor_vote_matrix.csv", sep=",", index_col=False)
    # data_clr = pd.read_csv("/home/komaldadhich/Desktop/test/multiclass_scatter/scm7/lab/Image_RGB.csv", sep=",",
    #                            index_col=False)
    # ''' To get legend colors and its labels'''
    # img = cv2.imread('/home/komaldadhich/Desktop/test/multiclass_scatter/scm7/sc7.png')
    # root = ET.parse('/home/komaldadhich/Desktop/test/multiclass_scatter/scm7/sc7_test.xml').getroot()

    # Now group stack heights based on it's catogery
    group_colors, group_leg_labels = get_legends(img, root)
    # group_colors = [[0,0,0]]
    # group_leg_labels = ['']
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())
    cord_list = {
            "x_val": [],
            "y_val": [],
            "CL": []
        }
    a = list(map(add, data_tensors["val1"], data_tensors["val2"]))
    amin, amax = min(a), max(a)
    for i, val in enumerate(a):
        a[i] = (val - amin) / (amax - amin)

    for i in range(X * Y):
        if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.4) and (a[i] > 0.003):
            if (data_tensors["X"][i] < (X - 5) and data_tensors["X"][i] > 5 and data_tensors["Y"][i] > 5 and data_tensors["Y"][i] < (Y - 5)):
                cord_list["x_val"].append(data_tensors["X"][i])
                cord_list["y_val"].append(data_tensors["Y"][i])
                cord_list["CL"].append((data_tensors["CL"][i]))

    cord_list['color'] = []
    for p in range(len(cord_list['x_val'])):
        cord_list['color'].append(data_clr.loc[(data_clr['X'] == cord_list['x_val'][p]) &
                                               (data_clr['Y'] == cord_list['y_val'][p])][['Red', 'Green', 'Blue']].values)


    data = np.array([cord_list['x_val'], cord_list['y_val']]).T

    # db = DBSCAN(eps=3, min_samples=2).fit(data) # scatt5
    # db = DBSCAN(eps=9, min_samples=4).fit(data)  # scatt7
    db = DBSCAN(eps=5, min_samples=4).fit(data) # scatt
    # db = DBSCAN(eps= 9 ,min_samples=5).fit(data) # scatt

    labels = db.labels_
    centers = []
    for i in (np.unique(labels)):
        if i<0:
            continue
        indexes = [id for id in range(len(labels)) if labels[id] == i]
        x = 0
        y = 0
        for k in indexes:
            x += cord_list['x_val'][k]
            y += cord_list['y_val'][k]
        centers += [[x // len(indexes), (y // len(indexes))]]

    img1 = img[::-1, :, :]
    # img1 = img
    group_id = []
    remove_ids = []
    # print(centers)
    # plot data with seaborn
    fig, ax = plt.subplots()
    # plt.suptitle("Critical points")
    # ax.axis('off')
    # plt.scatter(cord_list["x_val"], cord_list["y_val"], s=1)
    # plt.show()
    plt.scatter(cord_list["x_val"],  cord_list["y_val"], s=10, c='b')
    plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], s=10, c='r', alpha=0.7)
    plt.axis('off')
    plt.show()

    # group_id=[0 for i in range(len(centers))]


    # print("grp clr",group_colors)
    for i in range(len(centers)):
        if (img1[centers[i][1], centers[i][0]]).tolist() in group_colors:
            # print("color:", (img1[centers[i][1], centers[i][0]]).tolist())
            group_id += [group_colors.index(img1[centers[i][1], centers[i][0]].tolist())]
        else:
            color_pt = (img1[centers[i][1], centers[i][0]]).tolist()
            # print("clr:", color_pt)
            diff_r = (color_pt[0]-group_colors[0][0])**2
            diff_g = (color_pt[1]-group_colors[0][1])**2
            diff_b = (color_pt[2] - group_colors[0][2])**2

            diff_r1 = (color_pt[0] - group_colors[1][0])**2
            diff_g1 = (color_pt[1] - group_colors[1][1])**2
            diff_b1 = (color_pt[2] - group_colors[1][2])**2

            clr_diff_1= math.sqrt(diff_r + diff_g + diff_b)
            clr_diff_2= math.sqrt(diff_r1 + diff_g1 + diff_b1)

            if clr_diff_1>clr_diff_2:
                group_id += [0]
            else:
                group_id += [1]

    for j in range(len(group_colors)):
        group_colors[j] = np.array(group_colors[j][::-1])/255.0

    centers = np.delete(np.array(centers),remove_ids,axis=0)
    ''' Map pixels to original coordinates'''
    Xlabel,Ylabel,xbox_centers,ybox_centers = get_text_labels(img,root)
    # print("Y:",xbox_centers[0][0])
    if(isinstance(Ylabel[0], str) and Ylabel[0].isnumeric()):
        Ylabel = [int(i) for i in Ylabel if i.isnumeric()]
        for i in np.unique(Ylabel):
            id=[j for j, val in enumerate(Ylabel) if i==val]
            if(len(id)==2):
                if(ybox_centers[id[0]][1]<ybox_centers[id[1]][1]):
                    Ylabel[id[1]]*=-1
                    neg_ids=np.where(ybox_centers[:,1] > ybox_centers[id[1]][1])[0]
                else:
                    Ylabel[id[0]]*=-1
                    neg_ids=np.where(ybox_centers[:,1] > ybox_centers[id[0]][1])[0]
                for i in neg_ids:
                    Ylabel[i]*=-1


    if(isinstance(Xlabel[0], str) and Ylabel[0].isnumeric()):
        Xlabel = [int(i) for i in Xlabel if i.isnumeric()]
        for i in np.unique(Xlabel):
            id=[j for j, val in enumerate(Xlabel) if i==val]
            if(len(id)==2):
                if(xbox_centers[id[0]]>xbox_centers[id[1]]):
                    Xlabel[id[1]]*=-1
                    neg_ids=np.where(xbox_centers < xbox_centers[id[1]])[0]
                else:
                    Xlabel[id[0]]*=-1
                    neg_ids=np.where(xbox_centers < xbox_centers[id[0]])[0]
                for i in neg_ids:
                    Xlabel[i]*=-1
    print(Ylabel,Xlabel)

    '''normalize center to obtained labels'''
    centers = centers.astype(float)
    # xbox_centers, Xlabel = zip(*sorted(zip(xbox_centers, Xlabel)))
    xbox_centers, Xlabel = (list(xbox_centers),list(Xlabel))
    normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0][0]-xbox_centers[1][0])
    centers[:,0] = ((centers[:,0] - xbox_centers[0][0]) * normalize_scalex) + Xlabel[0]

    t = np.array(sorted(np.concatenate((ybox_centers, np.array([Ylabel]).T), axis=1), key=lambda x: x[1]))#, reverse= True))
    ybox_centers,Ylabel = (t[:,0:2],list(t[:,2]))
    normalize_scaley = (Ylabel[0]-Ylabel[1])/(ybox_centers[0][1]-ybox_centers[1][1])
    # centers[:,1] = ((centers[:,1] - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]
    centers[:, 1] = ((Y - centers[:, 1] - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]
    # normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0][0]-xbox_centers[1][0])
    # centers[:,0] = centers[:,0]*normalize_scalex
    # normalize_scaley =abs((Ylabel[0]-Ylabel[1])/(ybox_centers[0][1]-ybox_centers[1][1]))
    # centers[:,1] = centers[:,1]*normalize_scaley



    centers_x = []
    centers_y = []
    centers_clr=[]
    centers_labels=[]
    for i in centers:
        centers_x.append(i[0])
        centers_y.append(i[1])
    for i in range(len(group_id)):
        centers_clr.append(group_colors[group_id[i]])
        centers_labels.append(group_leg_labels[group_id[i]])

    '''Reconstruct Chart'''
    # for i in range(len(centers)):
    #     plt.scatter(centers[i][0], centers[i][1], c = group_colors[group_id[i]], label=group_leg_labels[group_id[i]])
    colors=ListedColormap(group_colors)
    # fig, ax = plt.subplots(figsize=(20,8))
    sc=plt.scatter(centers_x, centers_y, c=group_id, cmap=colors)
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.legend(handles=sc.legend_elements()[0], labels=group_leg_labels)

    plt.xlabel(get_xtitle(img, root))
    plt.ylabel(get_ytitle(img, root))
    plt.title(get_title(img, root))
    plt.tight_layout()
    plt.savefig(path + "reconstructed_" + image_name + ".png")
    plt.show()

    centers_x, centers_y, group_id = zip( *sorted(zip(centers_x, centers_y, group_id)) )

    # Writing data to CSV file
    L=[]
    for j in np.unique(group_id):
        L = L + [['X']+[group_leg_labels[j]]]
        L = L + [[centers_x[i], centers_y[i]] for i in range(len(centers_x)) if group_id[i] == j]

    with open(path+'data_'+image_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)

    print("Chart Reconstruction Done")
