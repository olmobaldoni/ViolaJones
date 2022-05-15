# Application for parsing the xml file and writing its data into the header file.

#===================================================================================#

import sys

# The xml.etree.ElementTree module implements a simple and efficient API for parsing and creating XML data.
import xml.etree.ElementTree as ET

# import cascade data by reading from a file
xml = sys.argv[1]
tree = ET.parse(xml)
root = tree.getroot()

#==================================initialize list==================================#

# list containing number of features for each stage
numFeatureStage_array = []

# list containing thresholds for each stage
thresholdStage_array = []

# list containing thresholds for each feature
tree_thresh_array = []

# list containing alpha1 values
alpha1_array = []

# list containing alpha2 values
alpha2_array = []

# list containing number of rectangles per feature
num_rectangles_array = []

# list containing rectangles' coordinates for each feature
rectangles_array = []

# list containing the weight of each rectangle that makes a feature
weights_array = []

#=========================iterate inside the xml file===============================#

# detector's height
height_window = root[0][2].text
height_window = int(height_window)

# detector's width
width_window = root[0][3].text
width_window = int(width_window)

# number of stages
numStages = root[0][6].text
numStages = int(numStages)

# for each stage, read the number of features and the threshold value
for i in range(numStages):
    numFeatureStage = root[0][7][i][0].text
    numFeatureStage = int(numFeatureStage)
    thresholdStage = root[0][7][i][1].text
    thresholdStage = float(thresholdStage)

    # multiply by 256 and convert the obtained value to an integer
    thresholdStage = int(thresholdStage*256) 

    # write the values in the respective lists
    numFeatureStage_array.append(numFeatureStage)
    thresholdStage_array.append(thresholdStage)

    # for each feature read "internalNodes", which contains: the corresponding index "rects" and the threshold value of the feature itself, and read "leafValues" which contains the values of alpha1 and alpha2 relative to the feature
    for j in range(numFeatureStage):
        internalNodes = root[0][7][i][2][j][0].text
        leafValues = root[0][7][i][2][j][1].text

        temp1 = internalNodes.split(" ")
        temp2 = leafValues.split(" ")

        # read the index "rects" in "internalNodes"
        rects_index = int(temp1[14])

        # read the feature threshold in "internalNodes"
        thresholdFeature = float(temp1[15])
        thresholdFeature = int(thresholdFeature*4096)

        # write the values in the respective lists
        tree_thresh_array.append(thresholdFeature)

        # read the values of alpha1 and alpha2 in "leafValues"
        alpha1 = float(temp2[12])
        alpha1 = int(alpha1*256)
        alpha2 = float(temp2[13])
        alpha2 = int(alpha2*256)

        # write the values in the respective lists
        alpha1_array.append(alpha1)
        alpha2_array.append(alpha2)


        # use the index obtained to find the corresponding "rects" in the xml

        # number of rectangles for each feature
        num_rectangles = len(list(root[0][8][rects_index][0].iter("_")))
        num_rectangles_array.append(num_rectangles)

        # for each rectangle read the respective coordinates and the respective weight
        for k in range(num_rectangles):
            a = root[0][8][rects_index][0][k].text
            b = a[:len(a) - 1]
            x = b.split()
            weights_array.append(int(x[-1]))
            lenght = len(x)
            for l in range(lenght - 1):
                rectangles_array.append(int(x[l]))
        

#===========================print data in the terminal==============================#

# print('numFeatureStage_array: ', numFeatureStage_array)
# print('\n')
# print('thresholdStage_array: ', thresholdStage_array)
# print('\n')
# print('tree_thresh_array: ', tree_thresh_array)
# print('\n')
# print('alpha1_array: ', alpha1_array)
# print('\n')
# print('alpha2_array: ', alpha2_array)
# print('\n')
# print('num_rectangles_array: ', num_rectangles_array)
# print('\n')
# print('weights_array: ', weights_array)
# print('\n')
# print('rectangles_array', rectangles_array)
# print('\n')
        
#========================write data inside the header file==========================#


f = open("converted_cascade.h","w")

f.write("const int insect_window_h = ")
f.write(str(height_window))
f.write(";")

f.write("\nconst int insect_window_w = ")
f.write(str(width_window))
f.write(";")

f.write("\nconst int insect_n_stages = ")
f.write(str(numStages))
f.write(";")

f.write("\nconst uint8_t insect_stages_array[] = { ")
for i in range(len(numFeatureStage_array)):
    f.write(str(numFeatureStage_array[i]))
    if i < (len(numFeatureStage_array) - 1):
        f.write(", ")
    else:
        f.write("};")

f.write("\nconst int16_t insect_stages_thresh_array[] = { ")
for i in range(len(thresholdStage_array)):
    f.write(str(thresholdStage_array[i]))
    if i < (len(thresholdStage_array) - 1):
        f.write(", ")
    else:
        f.write("};")

f.write("\nconst int16_t insect_node_thresh_array[] = { ")
for i in range(len(tree_thresh_array)):
    f.write(str(tree_thresh_array[i]))
    if i < (len(tree_thresh_array) - 1):
        f.write(", ")
    else:
        f.write("};")

f.write("\nconst int16_t insect_alpha1_array[] =  { ")
for i in range(len(alpha1_array)):
    f.write(str(alpha1_array[i]))
    if i < (len(alpha1_array) - 1):
        f.write(", ")
    else:
        f.write("};")

f.write("\nconst int16_t insect_alpha2_array[] = { ")
for i in range(len(alpha2_array)):
    f.write(str(alpha2_array[i]))
    if i < (len(alpha2_array) - 1):
        f.write(", ")
    else:
        f.write("};")

f.write("\nconst int8_t insect_num_rectangles_array[] = { ")
for i in range(len(num_rectangles_array)):
    f.write(str(num_rectangles_array[i]))
    if i < (len(num_rectangles_array) - 1):
        f.write(", ")
    else:
        f.write("};")

f.write("\nconst int8_t insect_weights_array[] = { ")
for i in range(len(weights_array)):
    f.write(str(weights_array[i]))
    if i < (len(weights_array) - 1):
        f.write(", ")
    else:
        f.write("};")

f.write("\nconst int8_t insect_rectangles_array[] = { ")
for i in range(len(rectangles_array)):
    f.write(str(rectangles_array[i]))
    if i < (len(rectangles_array) - 1):
        f.write(", ")
    else:
        f.write("};")

f.close()







        

