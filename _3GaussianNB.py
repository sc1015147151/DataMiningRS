#!/usr/bin/env python
# coding: utf-8

# In[22]:


import gdal
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical  
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
 


# In[2]:


lab_name=["./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart0.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart1.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart2.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart3.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart4.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart5.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart6.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart7.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart8.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1_label.tifpart9.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart0.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart1.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart2.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart3.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart4.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart5.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart6.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart7.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart8.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2_label.tifpart9.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart0.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart1.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart2.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart3.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart4.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart5.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart6.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart7.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart8.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2_label.tifpart9.csv",]
tif_name=["./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart0.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart1.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart2.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart3.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart4.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart5.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart6.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart7.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart8.csv",
 "./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tifpart9.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart0.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart1.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart2.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart3.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart4.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart5.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart6.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart7.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart8.csv",
 "./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tifpart9.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart0.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart1.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart2.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart3.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart4.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart5.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart6.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart7.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart8.csv",
 "./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tifpart9.csv",]


# In[3]:


import random
indx=  [k for k in range(30)]    
random.shuffle(indx)
#1.打乱文件的索引顺序，这样就能乱序训练了
rslt1=[]
rslt2=[]
rslt3=[]


# In[6]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
X=np.loadtxt(tif_name[0])
y =np.loadtxt(lab_name[0])#.reshape(-1, 1)
Xt=np.loadtxt(tif_name[29])
yt=np.loadtxt(lab_name[29])#.reshape(-1, 1)
clf.partial_fit(X, y,classes=np.array([[0],[1],[2],[3],[4],[5]]))
scr=clf.score(Xt,yt)
rslt3.append(scr)
for n in tqdm(indx[1:-15]) :
    
    X=np.loadtxt(tif_name[n])
    y =np.loadtxt(lab_name[n])#.reshape(-1, 1)
    
    Xt=np.loadtxt(tif_name[29-n])
    yt=np.loadtxt(lab_name[29-n])#.reshape(-1, 1)
    
    clf.partial_fit(X, y )
    scr=clf.score(Xt,yt)
    rslt3.append(scr)
print(rslt3)
 

# In[6]:


data1 = gdal.Open("./GID_samples/GF2_PMS1__L1A0001680858-MSS1.tif")
im_width = data1.RasterXSize #栅格矩阵的列数
im_height = data1.RasterYSize #栅格矩阵的行数
tif1=data1.ReadAsArray(0,0,im_width,im_height)
tif1=(cv2.merge(tif1)).reshape(-1,4)
pre_y1=clf.predict(tif1)
pre_y2=np.zeros((6800*7200, 3))
for i in tqdm(range(len(pre_y1))):#3.将像素标签转化为字符标签
        if pre_y1[i]==2:
            pre_y2[i]=[0, 255,   0]
        elif  pre_y1[i]==1:
            pre_y2[i]=[255,    0,    0]
        elif  pre_y1[i]==3:
            pre_y2[i]=[0,    255,  255]
        elif  pre_y1[i]==4:
            pre_y2[i]=[255, 255,     0]
        elif  pre_y1[i]==5:
            pre_y2[i]=[0,      0,  255]
        else:
            pre_y2[i]=[0,0,0]
pre_y3=pre_y2.reshape(6800,7200, 3)
 
b, g, r = cv2.split(pre_y3)
pre_y4 = cv2.merge([r, g, b])
cv2.imwrite('_3GF2_PMS1__L1A0001680858-MSS1_label.png',pre_y4)


# In[26]:


data1 = gdal.Open("./GID_samples/GF2_PMS2__L1A0000607677-MSS2.tif")
im_width = data1.RasterXSize #栅格矩阵的列数
im_height = data1.RasterYSize #栅格矩阵的行数
tif1=data1.ReadAsArray(0,0,im_width,im_height)
tif1=(cv2.merge(tif1)).reshape(-1,4)
pre_y1=clf.predict(tif1)
pre_y2=np.zeros((6800*7200, 3))
for i in tqdm(range(len(pre_y1))):#3.将像素标签转化为字符标签
        if pre_y1[i]==2:
            pre_y2[i]=[0, 255,   0]
        elif  pre_y1[i]==1:
            pre_y2[i]=[255,    0,    0]
        elif  pre_y1[i]==3:
            pre_y2[i]=[0,    255,  255]
        elif  pre_y1[i]==4:
            pre_y2[i]=[255, 255,     0]
        elif  pre_y1[i]==5:
            pre_y2[i]=[0,      0,  255]
        else:
            pre_y2[i]=[0,0,0]
pre_y3=pre_y2.reshape(6800,7200, 3)
 
b, g, r = cv2.split(pre_y3)
pre_y4 = cv2.merge([r, g, b])
cv2.imwrite('_3GF2_PMS2__L1A0000607677-MSS2_label.png',pre_y4)


# In[28]:


data1 = gdal.Open("./GID_samples/GF2_PMS2__L1A0000607681-MSS2.tif")
im_width = data1.RasterXSize #栅格矩阵的列数
im_height = data1.RasterYSize #栅格矩阵的行数
tif1=data1.ReadAsArray(0,0,im_width,im_height)
tif1=(cv2.merge(tif1)).reshape(-1,4)
pre_y1=clf.predict(tif1)
pre_y2=np.zeros((6800*7200, 3))
for i in tqdm(range(len(pre_y1))):#3.将像素标签转化为字符标签
        if pre_y1[i]==2:
            pre_y2[i]=[0, 255,   0]
        elif  pre_y1[i]==1:
            pre_y2[i]=[255,    0,    0]
        elif  pre_y1[i]==3:
            pre_y2[i]=[0,    255,  255]
        elif  pre_y1[i]==4:
            pre_y2[i]=[255, 255,     0]
        elif  pre_y1[i]==5:
            pre_y2[i]=[0,      0,  255]
        else:
            pre_y2[i]=[0,0,0]
pre_y3=pre_y2.reshape(6800,7200, 3)
 
b, g, r = cv2.split(pre_y3)
pre_y4 = cv2.merge([r, g, b])
cv2.imwrite('_3GF2_PMS2__L1A0000607681-MSS2_label.png',pre_y4)


# In[ ]:





# In[ ]:




