# Application for writing the bg.txt file required for training in the opencv_traincascade.
# Writes the path of each negative image inside the bg.txt file.

#===================================================================================#

import sys, os

a = os.getcwd()
b = "model_training/bg.txt"
c = os.path.join(a,b)

f = open(c,"w")

address = "neg/img"
termination = ".jpg"

x = int(sys.argv[1])

for i in range (1,x + 1):
    index = str(i)
    temp = address + index + termination
    f.write(temp)
    f.write("\n")

f.close()




