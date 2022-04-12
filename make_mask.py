import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os
import sys

hand1 = np.zeros((21, 2))
hand2 = np.zeros((21, 2))

# 畫骨架的線


def draw_line(hand, num1, num2, c):
    for i in range(num1, num2):
        x_label = [hand[i, 0], hand[i + 1, 0]]
        y_label = [hand[i, 1], hand[i + 1, 1]]
        plt.plot(x_label, y_label, color=c)

# 畫骨架、標點


def draw_skeleton(img, length):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.scatter(hand1[:, 0], hand1[:, 1], c='r')
    draw_line(hand1, 0, 4, 'lightgreen')
    draw_line(hand1, 5, 8, 'lightgreen')
    draw_line(hand1, 9, 12, 'lightgreen')
    draw_line(hand1, 13, 16, 'lightgreen')
    draw_line(hand1, 17, 20, 'lightgreen')
    plt.plot([hand1[0, 0], hand1[5, 0]], [
             hand1[0, 1], hand1[5, 1]], color='lightgreen')
    plt.plot([hand1[0, 0], hand1[17, 0]], [
             hand1[0, 1], hand1[17, 1]], color='lightgreen')
    plt.plot([hand1[5, 0], hand1[9, 0]], [
             hand1[5, 1], hand1[9, 1]], color='lightgreen')
    plt.plot([hand1[9, 0], hand1[13, 0]], [
             hand1[9, 1], hand1[13, 1]], color='lightgreen')
    plt.plot([hand1[13, 0], hand1[17, 0]], [
             hand1[13, 1], hand1[17, 1]], color='lightgreen')
    for i in range(21):
        plt.annotate(str(i), (hand1[i, 0], hand1[i, 1]), color='white')
    if length == 42:
        plt.scatter(hand2[:, 0], hand2[:, 1], c='b')
        draw_line(hand2, 0, 4, 'yellow')
        draw_line(hand2, 5, 8, 'yellow')
        draw_line(hand2, 9, 12, 'yellow')
        draw_line(hand2, 13, 16, 'yellow')
        draw_line(hand2, 17, 20, 'yellow')
        plt.plot([hand2[0, 0], hand2[5, 0]], [
                 hand2[0, 1], hand2[5, 1]], color='yellow')
        plt.plot([hand2[0, 0], hand2[17, 0]], [
                 hand2[0, 1], hand2[17, 1]], color='yellow')
        plt.plot([hand2[5, 0], hand2[9, 0]], [
                 hand2[5, 1], hand2[9, 1]], color='yellow')
        plt.plot([hand2[9, 0], hand2[13, 0]], [
                 hand2[9, 1], hand2[13, 1]], color='yellow')
        plt.plot([hand2[13, 0], hand2[17, 0]], [
                 hand2[13, 1], hand2[17, 1]], color='yellow')
        for i in range(21):
            plt.annotate(str(i), (hand2[i, 0], hand2[i, 1]), color='white')
    plt.imshow(image_rgb)
    plt.show()

# 畫mask、dilate、儲存圖片


def make_mask(img, length, store_location, store_name):
    if length == 21:
        hand_1 = np.array(hand1, dtype=np.int32)
        filled = cv2.fillPoly(img, pts=[hand_1], color=(255, 255, 255))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img_rgb)
        fig1 = plt.gcf()
        # plt.show()
        fig1.savefig(store_location + '\\' + store_name, bbox_inches='tight',
                     transparent=True, pad_inches=0)
    elif length == 42:
        hand_1 = np.array(hand1, dtype=np.int32)
        hand_2 = np.array(hand2, dtype=np.int32)
        filled = cv2.fillPoly(img, pts=[hand_1], color=(
            255, 255, 255)) + cv2.fillPoly(img, pts=[hand_2], color=(255, 255, 255))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img_rgb)
        fig1 = plt.gcf()
        # plt.show()
        fig1.savefig(store_location + '\\' + store_name, bbox_inches='tight',
                     transparent=True, pad_inches=0)
    img1_loc = store_location + '\\' + store_name
    img1 = cv2.imread(img1_loc)
    kernal = np.ones((2, 2), np.uint8)
    dilate = cv2.dilate(img1, kernal, iterations=1)
    cv2.imwrite(store_location + '\\' + store_name, dilate)

# 處理點


def json_data_segmentation(data):
    length = len(data)
    if length == 21:
        for i in range(length):
            hand1[i][0] = data[i]['x']
            hand1[i][1] = data[i]['y']
    elif length == 42:
        for i in range(0, 21):
            hand1[i][0] = data[i]['x']
            hand1[i][1] = data[i]['y']
        for i in range(21, 42):
            hand2[i-21][0] = data[i]['x']
            hand2[i-21][1] = data[i]['y']


'''
frames = os.listdir('.\\frames')
for frame in frames:
    for img in os.listdir('.\\frames\\' + frame):
        image = cv2.imread(f'.\\frames\\{frame}\\{img}')

labels = os.listdir('.\\label_landmark')
for label in labels:
    for js in os.listdir('.\\label_landmark\\' + label):
        with open()
'''

# frames : frames中的資料夾位置list
# labels : label_landmark中的資料夾位置list
# frame : 圖片位置list
# label : json檔list


frames = os.listdir('.\\frames')
labels = os.listdir('.\\label_landmark')
for i in range(0, len(frames)):
    frame = os.listdir('.\\frames\\' + frames[i])
    label = os.listdir('.\\label_landmark\\' + labels[i])
    save_loc = '.\\output_mask\\' + frames[i]
    for j in range(0, len(frame)):
        save_name = frame[j]
        origin_img = cv2.imread('.\\frames\\' + frames[i] + '\\' + frame[j])
        with open('.\\label_landmark\\' + labels[i] + '\\' + label[j]) as f:
            json_data = json.load(f)
        json_data_segmentation(json_data)
        make_mask(origin_img, len(json_data), save_loc, save_name)

'''
img_loc = '.\\frames\\00000\\00023.png'
json_loc = ".\\label_landmark\\00000\\00023.json"
# 讀取圖片 laod image
origin_img = cv2.imread(img_loc)


with open(json_loc) as f:
    json_data = json.load(f)
# print(json_data)
json_data_segmentation(json_data)
# print(hand1)
# print(hand2)

# draw_skeleton(origin_img, len(json_data))

# make_mask(origin_img, len(json_data))
'''
