import csv
import numpy as np
import cv2


# image test
image = "test_implem_a.jpg"

# xml cascade
cascade = "cascade20.xml"

# input image
img = cv2.imread(image, cv2.IMREAD_UNCHANGED)


# openCV detection stuff
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

insect_classifier = cv2.CascadeClassifier(cascade)

insects = insect_classifier.detectMultiScale(gray, 1.2, 1)

for (x,y,w,h) in insects:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 1) #green rectangle

print ('openCV detection:')
print(insects)
print('\n')


# stm32 detection stuff
rects = []

with open('20stageDetections.txt','r') as f:
    for l in f:
        row = l.split()
        rects.append([int(row[0]), int(row[1]), int(row[2]),int(row[3])])

rects,weights = cv2.groupRectangles(rects, 1, 0.4)

for (x,y,w,h) in rects:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 1) #blue rectangle

print('stm32 detection:')
print(rects)


#resize output image (bigger)
resized = cv2.resize(img, (100,100))


#show image
cv2.imshow('Insect detection', resized)

cv2.waitKey(0)

cv2.destroyAllWindows()