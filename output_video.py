import cv2
import numpy as np
import glob
import os

masks = os.listdir('.\\used\\output')
for i in range(0, len(masks)):
    img_array = []
    for filename in glob.glob('.\\used\\output\\' + masks[i] + '\\*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        '.\\videos\\' + 'mask_' + masks[i] + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
