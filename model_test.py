# Application to test the trained cascade.

#===========================================================#
import sys
import numpy as np
import cv2

cascade = sys.argv[1]
image = "testImages/" + sys.argv[2]

img = cv2.imread(image, cv2.IMREAD_UNCHANGED)


insect_classifier = cv2.CascadeClassifier(cascade)


insects = insect_classifier.detectMultiScale(img, 1.2, 5)

# iterate through insects array and draw a rectangle over each insect in insects
for (x,y,w,h) in insects:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Insect detection', img)

cv2.waitKey(0)

cv2.destroyAllWindows()


