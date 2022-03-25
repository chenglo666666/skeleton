import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('.\\output_mask\\*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


out = cv2.VideoWriter('mask.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
