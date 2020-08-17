import numpy as np
import cv2 
from matplotlib import pyplot as plt

ipath = "./allfull.jpg"
lpath = "./label.npy"


# last of v1 : 46232
# last of v2 : 91164


class MyExplorer():
    def __init__(self,img_path,label_path,fps=25,size=(40,40)):
        self.img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        self.l = np.load(label_path)
        self.fps = fps
        self.size = size
    def subimg(self,i):
        subimg = np.empty(self.size)
        #print(subimg.shape)
        
        m = i // (self.fps*60) 
        
        f = i % (self.fps*60)
        idx = (m*self.size[0],f*self.size[1])
        
        subimg = self.img[idx[0]:idx[0]+self.size[0],idx[1]:idx[1]+self.size[1]]
        return subimg
    def label(self,i):
        return self.l[i]
    
    def split_single(self):
        # first two videos will be train set and the last video is valid set
        train_x=[]
        train_y=[]
        valid_x=[]
        valid_y=[]
        for i in range (91165):
            train_x.append(self.subimg(i))
            train_y.append(self.label(i))
        for i in range(91165,134731):
            valid_x.append(self.subimg(i))
            valid_y.append(self.label(i))
        return train_x,train_y,valid_x,valid_y
    def split_double(self):
         # first two videos will be train set and the last video is valid set
        train_x=[]
        train_y=[]
        valid_x=[]
        valid_y=[]
        for i in range (46232):
            dimg = np.empty([self.size[0],self.size[1]*2],dtype = "uint8")
            dimg[0:self.size[0],0:self.size[1]] = self.subimg(i)
            dimg[0:self.size[0],self.size[1]:self.size[1]*2] = self.subimg(i+1)
            dlabel = self.label(i) * self.label(i+1)
            train_x.append(dimg)
            train_y.append(dlabel)
        for i in range (46233,91164):
            dimg = np.empty([self.size[0],self.size[1]*2],dtype = "uint8")
            dimg[0:self.size[0],0:self.size[1]] = self.subimg(i)
            dimg[0:self.size[0],self.size[1]:self.size[1]*2] = self.subimg(i+1)
            dlabel = self.label(i) * self.label(i+1)
            train_x.append(dimg)
            train_y.append(dlabel)
        for i in range(91165,134730):
            dimg = np.empty([self.size[0],self.size[1]*2],dtype = "uint8")
            dimg[0:self.size[0],0:self.size[1]] = self.subimg(i)
            dimg[0:self.size[0],self.size[1]:self.size[1]*2] = self.subimg(i+1)
            dlabel = self.label(i) * self.label(i+1)
            valid_x.append(dimg)
            valid_y.append(dlabel)
        return train_x,train_y,valid_x,valid_y


#example
#dataset = MyExplorer(ipath,lpath)

#X,Y,x,y = dataset.split_single()
#img = X[5]
#plt.imshow(img)
#print(Y[5])


#XX,YY,xx,yy = dataset.split_double()
#dimg = XX[5]
#plt.imshow(dimg)
#print(YY[5])





