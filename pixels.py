# Small application to obtain the values of each pixel that compose an image

#=========================================================#
import sys
from PIL import Image
import numpy as np

image = sys.argv[1]
x = "testImages/" + image

im = Image.open(x)
pixels = list(im.getdata())

print(pixels)

matrix = np.array(im.getdata()).reshape(im.size)

print(matrix)