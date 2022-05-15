# Application to print the detections and iou values

from collections import namedtuple
import numpy as np
import cv2


#=======================================================================================#
# Print the various detections obtained on the terminal


image = "testImages/test_implem_a.jpg"

img = cv2.imread(image, cv2.IMREAD_UNCHANGED)


insect_classifier = cv2.CascadeClassifier("cascade15.xml")


insects = insect_classifier.detectMultiScale(img, 1.2, 5)


for i in range (len(insects)):
    print(insects[i])

#=======================================================================================#
# Print the iou value


# # define the 'Detection' object
# detection = namedtuple("Detection", ["image_path", "gt", "pred"])

# # image_path : the path to our input image
# # gt : the ground-truth bounding box
# # pred : the predicted bounding box from our model


# def bb_intersection_over_union(boxA, boxB):
#     # determine the (x, y) - coordinates of the intersection rectangle
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

#     # compute the area of intersection rectangle
#     interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
#     if interArea == 0:
#         return 0
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = abs((boxA[2]) * (boxA[3]))
#     boxBArea = abs((boxB[2]) * (boxB[3]))

#     # compute the area of the union of the rectangles
#     unionArea = float(boxAArea + boxBArea - interArea)
#     # compute the intersection over union 
    
#     iou = interArea / unionArea

#     # return the intersection over union value
#     return iou



# # define the list of example detection
# examples = [detection("testImages/f.jpg", [ 210, 174, 19, 14], [ 209, 168, 22, 22])]

# # loop over the example detections
# for detection in examples:
#     #load the image
#     image = cv2.imread(detection.image_path)

#     # draw the ground-truth bounding box along the predicted boundinx box
#     cv2.rectangle(image, tuple(detection.gt[:2]), (detection.gt[0] + detection.gt[2], detection.gt[1] + detection.gt[3]), (0, 255, 0), 2)
#     cv2.rectangle(image, tuple(detection.pred[:2]), (detection.pred[0] + detection.pred[2], detection.pred[1] + detection.pred[3]), (0, 0, 255), 2)

#     #compute the intersection over union and display it
#     iou = bb_intersection_over_union(detection.gt, detection.pred)
#     cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     print("{}: {:.4f}".format(detection.image_path, iou))

#     #show the output image
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)




