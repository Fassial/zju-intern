"""
Created on July 22 09:12, 2020

@author: fassial
"""
# get image
from skimage import io
# local dep
import sys
sys.path.append("../")
from hog import hog

EXAMPLE_FILE = "./example.jpg"
RESULT_FILE = "./example-hog.jpg"

"""
test:
    test hog
    @params:
        None
    @rets:
        None
"""
def test():
    image = io.imread(EXAMPLE_FILE, as_gray = True)
    normalised_blocks, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), visualize=True)
    io.imsave(RESULT_FILE, hog_image)

if __name__ == "__main__":
    test()
