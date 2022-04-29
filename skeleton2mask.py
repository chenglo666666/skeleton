import os
import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import random

# draw lines for skeleton


def draw_line(hand, num1, num2, c):
    for i in range(num1, num2):
        x_label = [hand[i, 0], hand[i + 1, 0]]
        y_label = [hand[i, 1], hand[i + 1, 1]]
        plt.plot(x_label, y_label, color=c)

# draw skeleton


def draw_skeleton(img, length, hand_1, hand_2):
    plt.clf()
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.scatter(hand_1[:, 0], hand_1[:, 1], c='r')
    draw_line(hand_1, 0, 4, 'lightgreen')
    draw_line(hand_1, 5, 8, 'lightgreen')
    draw_line(hand_1, 9, 12, 'lightgreen')
    draw_line(hand_1, 13, 16, 'lightgreen')
    draw_line(hand_1, 17, 20, 'lightgreen')
    plt.plot([hand_1[0, 0], hand_1[5, 0]], [
             hand_1[0, 1], hand_1[5, 1]], color='lightgreen')
    plt.plot([hand_1[0, 0], hand_1[17, 0]], [
             hand_1[0, 1], hand_1[17, 1]], color='lightgreen')
    plt.plot([hand_1[5, 0], hand_1[9, 0]], [
             hand_1[5, 1], hand_1[9, 1]], color='lightgreen')
    plt.plot([hand_1[9, 0], hand_1[13, 0]], [
             hand_1[9, 1], hand_1[13, 1]], color='lightgreen')
    plt.plot([hand_1[13, 0], hand_1[17, 0]], [
             hand_1[13, 1], hand_1[17, 1]], color='lightgreen')
    '''
    for i in range(21):
        plt.annotate(str(i), (hand1[i, 0], hand1[i, 1]), color='white')
    '''
    if length == 42:
        plt.scatter(hand_2[:, 0], hand_2[:, 1], c='b')
        draw_line(hand_2, 0, 4, 'yellow')
        draw_line(hand_2, 5, 8, 'yellow')
        draw_line(hand_2, 9, 12, 'yellow')
        draw_line(hand_2, 13, 16, 'yellow')
        draw_line(hand_2, 17, 20, 'yellow')
        plt.plot([hand_2[0, 0], hand2[5, 0]], [
                 hand_2[0, 1], hand_2[5, 1]], color='yellow')
        plt.plot([hand_2[0, 0], hand_2[17, 0]], [
                 hand_2[0, 1], hand_2[17, 1]], color='yellow')
        plt.plot([hand_2[5, 0], hand_2[9, 0]], [
                 hand_2[5, 1], hand_2[9, 1]], color='yellow')
        plt.plot([hand_2[9, 0], hand_2[13, 0]], [
                 hand_2[9, 1], hand_2[13, 1]], color='yellow')
        plt.plot([hand_2[13, 0], hand_2[17, 0]], [
                 hand_2[13, 1], hand_2[17, 1]], color='yellow')
        '''
        for i in range(21):
            plt.annotate(str(i), (hand2[i, 0], hand2[i, 1]), color='white')
        '''
    plt.axis('off')
    plt.imshow(image_rgb)
    figure = plt.gcf()
    return figure


# random noises


def random_noises(hand):
    for i in range(0, len(hand)):
        x_or_y = random.randint(0, 1)
        shift_distance = random.randint(-1, 1)
        if x_or_y == 0:
            hand[i][0] += shift_distance
        else:
            hand[i][1] += shift_distance
    return hand


# convex hull algorithm implementation


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def convex_hull(points, length):
    min = 0
    for i in range(1, length):
        if points[i].x < points[min].x:
            min = i
        elif points[i].x == points[min].x:
            if points[i].y > points[min].y:
                min = i
    most_left_index = min
    hull = []
    p = most_left_index
    q = 0
    while True:
        hull.append(p)
        q = (p + 1) % length
        for i in range(length):
            val = (points[i].y - points[p].y) * (points[q].x - points[i].x) - \
                   (points[i].x - points[p].x) * (points[q].y - points[i].y)
            if val < 0:
                q = i
        p = q
        if p == most_left_index:
            break
    return hull


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


landmarkRoot = '.\\label_landmark'
root_multiColorSkeleton = '.\\frames'
outputMask = '.\\output'
os.makedirs(outputMask, exist_ok=True)


hand1 = np.zeros((21, 2))
hand2 = np.zeros((21, 2))


for video in sorted(os.listdir(root_multiColorSkeleton)):
    v = root_multiColorSkeleton + '\\' + video
    assert os.path.exists(v), 'video path is incorrect!'
    framesAmount = len(os.listdir(v))
    # paths
    '''
    maskPath = f'{outputMask}/mask'
    os.makedirs(maskPath, exist_ok=True)
    '''
    cropPath = f'{outputMask}/cropped'
    os.makedirs(cropPath, exist_ok=True)
    '''
    visualizePath = f'{outputMask}/visualize'
    os.makedirs(visualizePath, exist_ok=True)
    video_visPath = f'{outputMask}/vis_detectedPerformance'
    os.makedirs(video_visPath, exist_ok=True)
    '''
    convex_hull_path = f'{outputMask}/ConvexHull'
    os.makedirs(convex_hull_path, exist_ok=True)
    convex_hull_mask_path = f'{outputMask}/ConvexHullMask'
    os.makedirs(convex_hull_mask_path, exist_ok=True)
    noise_visPath = f'{outputMask}/noises_detectedPerformance'
    os.makedirs(noise_visPath, exist_ok=True)
    noise_visualize_path = f'{outputMask}/noise_visualize'
    os.makedirs(noise_visualize_path, exist_ok=True)
    # visualize noises video
    vid = cv2.VideoWriter_fourcc(*'mp4v')
    video_noise = cv2.VideoWriter(
        f'{noise_visPath}/noise_{video}.mp4', vid, 30, (244, 244))
    # visualize video
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_detectPerformance = cv2.VideoWriter(
        f'{video_visPath}/detected_{video}.mp4', fourcc, 30, (224, 224))
    '''
    frame_idx = 0
    for frame in sorted(os.listdir(v)):
        image = cv2.imread(f'{v}/{frame}')
        noise_img = image.copy()
        cropped = image.copy()
        convex_hull_cropped = image.copy()
        # mask = np.zeros(image.shape[:2], dtype=np.uint8)

        frame = frame.split('.')[0]
        jsonFile = f'{landmarkRoot}/{video}/{frame}.json'
        # assert os.path.exists(jsonFile), 'landmark file does not exist!!'s
        if os.path.exists(jsonFile):
            f = open(jsonFile, 'r')
            landmarks = json.load(f)
            # save points as numpy array
            json_data_segmentation(landmarks)

            if len(landmarks) == 21:
                hand_1 = np.array(hand1, dtype=np.int32)
                crop = np.zeros(image.shape)
                '''
                mask = cv2.fillPoly(mask, pts=[hand_1], color=(8, 8, 8))
                drawed = cv2.fillPoly(image, pts=[hand_1], color=(0, 255, 0))
                '''

                x = [a[0] for a in hand_1]
                y = [a[1] for a in hand_1]
                x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
                crop[y0:y1, x0:x1, 0] = cropped[y0:y1, x0:x1, 0]
                crop[y0:y1, x0:x1, 1] = cropped[y0:y1, x0:x1, 1]
                crop[y0:y1, x0:x1, 2] = cropped[y0:y1, x0:x1, 2]

                # do convex hull, return a numpy array with outline points
                points = []
                for i in range(0, 21):
                    points.append(Point(hand1[i][0], hand1[i][1]))
                hull = convex_hull(points, 21)
                outline1 = np.zeros((len(hull), 2))
                for i in range(len(hull)):
                    outline1[i][0] = hand1[hull[i]][0]
                    outline1[i][1] = hand1[hull[i]][1]

                # save convex hull's mask and convex hull's cropped images
                rows = convex_hull_cropped.shape[0]
                cols = convex_hull_cropped.shape[1]
                channels = convex_hull_cropped.shape[2]
                convex_hull_crop = np.zeros(
                    convex_hull_cropped.shape, dtype=np.uint8)
                coordinate = np.array(outline1, dtype=np.int32)
                ignore_mask_color = (255,)*channels
                convex_hull_crop = cv2.fillPoly(
                    convex_hull_crop, [coordinate], ignore_mask_color)
                convex_hull_img = cv2.bitwise_and(
                    convex_hull_cropped, convex_hull_crop)

                os.makedirs(f'{convex_hull_path}/{video}', exist_ok=True)
                cv2.imwrite(
                    f'{convex_hull_path}/{video}/{frame}.png', convex_hull_img)
                os.makedirs(f'{convex_hull_mask_path}/{video}', exist_ok=True)
                cv2.imwrite(
                    f'{convex_hull_mask_path}/{video}/{frame}.png', convex_hull_crop)

                # save noise images, and then save those images as video
                os.makedirs(f'{noise_visualize_path}/{video}', exist_ok=True)
                noise_1 = random_noises(hand1)
                noise_figure = draw_skeleton(noise_img, 21, noise_1, hand2)
                noise_figure.savefig(f'{noise_visualize_path}/{video}/{frame}.png',
                                     bbox_inches='tight', transparent=True, pad_inches=0)
                video_noise.write(cv2.imread(
                    f'{noise_visualize_path}/{video}/{frame}.png'))
                '''
                for i in hand_1:
                    # mask number set to 8
                    mask[i[1], i[0]] = 8
                # save mask
                os.makedirs(f'{maskPath}/{video}', exist_ok=True)
                cv2.imwrite(f'{maskPath}/{video}/{frame}.png', mask)
                '''
                # save cropped hands
                os.makedirs(f'{cropPath}/{video}', exist_ok=True)
                cv2.imwrite(f'{cropPath}/{video}/crop_{frame}.png', crop)
                '''
                # save crop
                cv2.putText(drawed, 'frame: {}'.format(frame_idx),
                            (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                img_rgb = cv2.cvtColor(drawed, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img_rgb)
                fig1 = plt.gcf()
                os.makedirs(f'{visualizePath}/{video}', exist_ok=True)
                fig1.savefig(f'{visualizePath}/{video}/{frame}.png',
                             bbox_inches='tight', transparent=True, pad_inches=0)
                video_detectPerformance.write(drawed)
                '''
            else:
                # Case: 2 hands
                hand_1 = np.array(hand1, dtype=np.int32)
                hand_2 = np.array(hand2, dtype=np.int32)
                crop = np.zeros(image.shape)
                '''
                mask = cv2.fillPoly(mask, pts=[hand_1], color=(8, 8, 8))
                mask = cv2.fillPoly(mask, pts=[hand_2], color=(8, 8, 8))

                drawed = cv2.fillPoly(image, pts=[hand_1], color=(0, 255, 0))
                drawed = cv2.fillPoly(image, pts=[hand_2], color=(0, 255, 0))
                '''
                # drawed = cv2.fillPoly(image, pts=[hand_1], color=(255, 255, 255)) + cv2.fillPoly(image, pts=[hand_2], color=(255, 255, 255))
                # crop 2 hands

                for hand in [hand_1, hand_2]:
                    x = [a[0] for a in hand]
                    y = [a[1] for a in hand]
                    '''
                    for i in hand:
                        # mask number set to 8
                        mask[i[1], i[0]] = 8
                    '''
                    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
                    crop[y0:y1, x0:x1, 0] = cropped[y0:y1, x0:x1, 0]
                    crop[y0:y1, x0:x1, 1] = cropped[y0:y1, x0:x1, 1]
                    crop[y0:y1, x0:x1, 2] = cropped[y0:y1, x0:x1, 2]

                # do convex hull
                # hand1
                points = []
                for i in range(0, 21):
                    points.append(Point(hand1[i][0], hand1[i][1]))
                hull = convex_hull(points, 21)
                outline1 = np.zeros((len(hull), 2))
                for i in range(len(hull)):
                    outline1[i][0] = hand1[hull[i]][0]
                    outline1[i][1] = hand1[hull[i]][1]
                # hand2
                points = []
                for i in range(0, 21):
                    points.append(Point(hand2[i][0], hand2[i][1]))
                hull = convex_hull(points, 21)
                outline2 = np.zeros((len(hull), 2))
                for i in range(len(hull)):
                    outline2[i][0] = hand2[hull[i]][0]
                    outline2[i][1] = hand2[hull[i]][1]

                # save convex hull's mask and convex hull's cropped images
                rows = convex_hull_cropped.shape[0]
                cols = convex_hull_cropped.shape[1]
                channels = convex_hull_cropped.shape[2]
                convex_hull_crop = np.zeros(
                    convex_hull_cropped.shape, dtype=np.uint8)
                coordinate1 = np.array(outline1, dtype=np.int32)
                coordinate2 = np.array(outline2, dtype=np.int32)
                ignore_mask_color = (255,)*channels
                convex_hull_crop = cv2.fillPoly(convex_hull_crop, [coordinate1], ignore_mask_color) + cv2.fillPoly(
                    convex_hull_crop, [coordinate2], ignore_mask_color)
                convex_hull_img = cv2.bitwise_and(
                    convex_hull_cropped, convex_hull_crop)

                os.makedirs(f'{convex_hull_path}/{video}', exist_ok=True)
                cv2.imwrite(
                    f'{convex_hull_path}/{video}/{frame}.png', convex_hull_img)
                os.makedirs(f'{convex_hull_mask_path}/{video}', exist_ok=True)
                cv2.imwrite(
                    f'{convex_hull_mask_path}/{video}/{frame}.png', convex_hull_crop)

                # save noise images, and then save those images as video
                os.makedirs(f'{noise_visualize_path}/{video}', exist_ok=True)
                noise_1 = random_noises(hand1)
                noise_2 = random_noises(hand2)
                noise_figure = draw_skeleton(noise_img, 42, noise_1, noise_2)
                noise_figure.savefig(f'{noise_visualize_path}/{video}/{frame}.png',
                                     bbox_inches='tight', transparent=True, pad_inches=0)
                video_noise.write(cv2.imread(
                    f'{noise_visualize_path}/{video}/{frame}.png'))

                '''
                # save mask
                os.makedirs(f'{maskPath}/{video}', exist_ok=True)
                cv2.imwrite(f'{maskPath}/{video}/{frame}.png', mask)
                '''
                # save crop hands
                os.makedirs(f'{cropPath}/{video}', exist_ok=True)
                cv2.imwrite(f'{cropPath}/{video}/crop_{frame}.png', crop)
                '''
                cv2.putText(drawed, 'frame: {}'.format(frame_idx),
                            (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                img_rgb = cv2.cvtColor(drawed, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img_rgb)
                fig1 = plt.gcf()
                os.makedirs(f'{visualizePath}/{video}', exist_ok=True)
                fig1.savefig(f'{visualizePath}/{video}/{frame}.png',
                             bbox_inches='tight', transparent=True, pad_inches=0)
                video_detectPerformance.write(drawed)
                '''
        '''
            kernel = np.ones((2, 2), np.uint8)
            dilate = cv2.dilate(image, kernel, iterations=1)

        else:
            # save mask
            os.makedirs(f'{maskPath}/{video}', exist_ok=True)
            cv2.imwrite(f'{maskPath}/{video}/{frame}.png', mask)
            # save black mask as crop hands
            os.makedirs(f'{cropPath}/{video}', exist_ok=True)
            cv2.imwrite(f'{cropPath}/{video}/crop_{frame}.png', mask)
            cv2.putText(image, 'notDetected_frame: {}'.format(
                frame_idx), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            # save video
            video_detectPerformance.write(image)
        '''
        frame_idx += 1
    '''
    # check mask amount is epual to frames amount
    if len(os.listdir(f'{maskPath}/{video}')) != framesAmount:
        assert 'mask amount is incorrect!'
    '''
    print(f'video: {video} is done.')
