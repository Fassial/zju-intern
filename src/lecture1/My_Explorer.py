

import numpy as np
import cv2 
#from matplotlib import pyplot as plt



ipath = "allfull.jpg"
lpath = "label.npy"



    
class MyExplorer():
    def __init__(self,img_path,label_path,fps=25,size=(40,40)):
        self.img = cv2.imread(img_path)
        self.l = np.load(label_path)
        self.fps = fps
        self.size = size
    def subimg(self,i):
        subimg = np.empty(self.size)
        #print(subimg.shape)
        m = i // (self.fps*60) 
        s = i % (self.fps*60) // self.fps 
        f = i % (self.fps*60) % self.fps
        idx = (m*self.size[0],(s*self.fps+f)*self.size[1])
        subimg = self.img[idx[0]:idx[0]+self.size[0],idx[1]:idx[1]+self.size[1]]
        return subimg
    def label(self,i):
        return self.l[i]


# example
dataset = MyExplorer(ipath,lpath)

img = dataset.subimg(79389)
label = dataset.label(79389)


plt.imshow(img)
plt.show()







