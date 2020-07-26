from .My_Explorer import MyExplorer
"""
read_image:
    
    @params:
        ipath(str)  : the name of folder with pictures
        lpath(str)  : the name of folder with labels
    @rets:
        image_data(list)  :  the data of image
        image_label(list) :  the label of image
"""
def read_image(ipath,lpath):
    image_data = []
    image_label = []
    dataset = MyExplorer(ipath,lpath)
    total_num =dataset.getTotal()
    for i in range(0,total_num):
        image_data.append(dataset.subimg(i))
        image_label.append(dataset.label(i))
    return image_data,image_label


