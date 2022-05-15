# Small application to obtain the values of each pixel that compose an image

#=========================================================#

from PIL import Image
import numpy as np
import cv2


img = cv2.imread("test_implem_c.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


pixels = []



for i in range (100):
    for j in range (100):
        pixels.append(gray[i][j])

f = open("pixels.txt","w")

f.write("\npixels[] = { ")
for i in range(len(pixels)):
    f.write(str(pixels[i]))
    if i < (len(pixels) - 1):
        f.write(", ")
    else:
        f.write("};")

f.close()
