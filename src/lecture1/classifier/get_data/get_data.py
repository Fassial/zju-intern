import glob
import cv2
import os

"""
read_image:
    
    @params:
        foldername(str) : the folder name with pictures to read
        isPositive(bool) : positive or negative
    @rets:
        image_data(list)  :  the data of image
        image_label(list) :  the label of image
"""
def read_image(foldername,isPositive = True):
    image_data = []
    image_label = []
    filename = os.path.join(foldername,'*')
    filename_iterator = glob.iglob(filename)
    for image_name in filename_iterator:
        image = cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)
        image_data.append(image)
        if isPositive == True:
            image_label.append(1)
        else:
            image_label.append(0)
    return image_data,image_label

