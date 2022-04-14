import os
import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

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

def get_framesAmount(path):
    return len(os.listdir(path))

d_t = ['test']

for dt in d_t:
    landmarkRoot = f'../mediapipe/data/MultiColor_skeleton_totalFrames/WLASL_100/{dt}/label_landmark'
    root_multiColorSkeleton = f'../mediapipe/data/Bbox_cropBySkeleton_totalFrames/WLASL_100/{dt}/frames'
    outputMask = f'../mediapipe/data/Hand_Mask_totalFrames/WLASL_100/{dt}'
    os.makedirs(outputMask, exist_ok=True)


    hand1 = np.zeros((21, 2))
    hand2 = np.zeros((21, 2))



    for video in sorted(os.listdir(root_multiColorSkeleton)):
        v = f'{root_multiColorSkeleton}/{video}'
        assert os.path.exists(v), 'video path is incorrect!'
        framesAmount = len(os.listdir(v))
        # paths 
        maskPath = f'{outputMask}/mask'
        os.makedirs(maskPath, exist_ok=True)
        cropPath = f'{outputMask}/cropped'
        os.makedirs(cropPath, exist_ok=True)
        visualizePath = f'{outputMask}/visualize'
        os.makedirs(visualizePath, exist_ok=True)
        video_visPath = f'{outputMask}/vis_detectedPerformance'
        os.makedirs(video_visPath, exist_ok=True)
        # visualize video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_detectPerformance = cv2.VideoWriter(f'{video_visPath}/detected_{video}.mp4', fourcc, 30, (224, 224))
        frame_idx = 0
        for frame in sorted(os.listdir(v)):
            image = cv2.imread(f'{v}/{frame}')
            cropped = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
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
                    mask = cv2.fillPoly(mask, pts=[hand_1], color=(8, 8, 8))
                    drawed = cv2.fillPoly(image, pts=[hand_1], color=(0, 255, 0))
                    
                    x = [a[0] for a in hand_1]
                    y = [a[1] for a in hand_1]
                    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
                    crop[y0:y1, x0:x1, 0] = cropped[y0:y1, x0:x1, 0]
                    crop[y0:y1, x0:x1, 1] = cropped[y0:y1, x0:x1, 1]
                    crop[y0:y1, x0:x1, 2] = cropped[y0:y1, x0:x1, 2]
                    for i in hand_1:
                        # mask number set to 8
                        mask[i[1], i[0]] = 8
                    # save mask
                    os.makedirs(f'{maskPath}/{video}', exist_ok=True)
                    cv2.imwrite(f'{maskPath}/{video}/{frame}.png', mask)
                    # save cropped hands
                    os.makedirs(f'{cropPath}/{video}', exist_ok=True)
                    cv2.imwrite(f'{cropPath}/{video}/crop_{frame}.png', crop)
                    # save crop
                    cv2.putText(drawed, 'frame: {}'.format(frame_idx), (0,50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                    img_rgb = cv2.cvtColor(drawed, cv2.COLOR_BGR2RGB)
                    plt.axis('off')
                    plt.imshow(img_rgb)
                    fig1 = plt.gcf()
                    os.makedirs(f'{visualizePath}/{video}', exist_ok=True)
                    fig1.savefig(f'{visualizePath}/{video}/{frame}.png', bbox_inches='tight', transparent=True, pad_inches=0)
                    video_detectPerformance.write(drawed)
                else:
                    # Case: 2 hands
                    hand_1 = np.array(hand1, dtype=np.int32)
                    hand_2 = np.array(hand2, dtype=np.int32)
                    crop = np.zeros(image.shape)
                    mask = cv2.fillPoly(mask, pts=[hand_1], color=(8, 8, 8))
                    mask = cv2.fillPoly(mask, pts=[hand_2], color=(8, 8, 8))
                    
                    drawed = cv2.fillPoly(image, pts=[hand_1], color=(0, 255, 0))
                    drawed = cv2.fillPoly(image, pts=[hand_2], color=(0, 255, 0))
                    
                    # drawed = cv2.fillPoly(image, pts=[hand_1], color=(255, 255, 255)) + cv2.fillPoly(image, pts=[hand_2], color=(255, 255, 255))
                    # crop 2 hands         
                    for hand in [hand_1, hand_2]:
                        x = [a[0] for a in hand]
                        y = [a[1] for a in hand]
                        for i in hand:
                            # mask number set to 8
                            mask[i[1], i[0]] = 8
                        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
                        crop[y0:y1, x0:x1, 0] = cropped[y0:y1, x0:x1, 0]
                        crop[y0:y1, x0:x1, 1] = cropped[y0:y1, x0:x1, 1]
                        crop[y0:y1, x0:x1, 2] = cropped[y0:y1, x0:x1, 2]
                    # save mask
                    os.makedirs(f'{maskPath}/{video}', exist_ok=True)
                    cv2.imwrite(f'{maskPath}/{video}/{frame}.png', mask)
                    # save crop hands
                    os.makedirs(f'{cropPath}/{video}', exist_ok=True)
                    cv2.imwrite(f'{cropPath}/{video}/crop_{frame}.png', crop)
                    cv2.putText(drawed, 'frame: {}'.format(frame_idx), (0,50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                    img_rgb = cv2.cvtColor(drawed, cv2.COLOR_BGR2RGB)
                    plt.axis('off')
                    plt.imshow(img_rgb)
                    fig1 = plt.gcf()
                    os.makedirs(f'{visualizePath}/{video}', exist_ok=True)
                    fig1.savefig(f'{visualizePath}/{video}/{frame}.png', bbox_inches='tight', transparent=True, pad_inches=0)
                    video_detectPerformance.write(drawed)
                    
                kernel = np.ones((2,2), np.uint8)
                dilate = cv2.dilate(image, kernel, iterations=1)
            else:
                # save mask
                os.makedirs(f'{maskPath}/{video}', exist_ok=True)
                cv2.imwrite(f'{maskPath}/{video}/{frame}.png', mask)
                # save black mask as crop hands
                os.makedirs(f'{cropPath}/{video}', exist_ok=True)
                cv2.imwrite(f'{cropPath}/{video}/crop_{frame}.png', mask)
                cv2.putText(image, 'notDetected_frame: {}'.format(frame_idx), (0,50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                # save video
                video_detectPerformance.write(image)
            frame_idx += 1
        # check mask amount is epual to frames amount
        if len(os.listdir(f'{maskPath}/{video}')) != framesAmount:
            assert 'mask amount is incorrect!'
        print(f'video: {video} is done.')
