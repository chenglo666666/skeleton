import os
import cv2
import json
import enum
import dataclasses
import numpy as np
from tqdm import tqdm
import multiprocessing
import mediapipe as mp
from typing import Tuple
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ---------mediaPipe related---------
WHITE_COLOR = (224, 224, 224)
_THICKNESS_WRIST_MCP = 3
_THICKNESS_FINGER = 2
_THICKNESS_DOT = -1
_RADIUS = 1
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
class HandLandmark(enum.IntEnum):
  """The 21 hand landmarks."""
  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Parameter adjust: https://blog.csdn.net/luozhichengaichenlei/article/details/117262688
DrawingSpec_point = mp_drawing.DrawingSpec((0, 0, 255), 1, 1) #color, thickness, radius
DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 2, 2) #color, thickness, radius


@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2

# Hand landmarks
_PALM_LANMARKS = (HandLandmark.WRIST, HandLandmark.THUMB_CMC,
                  HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP,
                  HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP)

_THUMP_LANDMARKS = (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP,
                    HandLandmark.THUMB_TIP)

_INDEX_FINGER_LANDMARKS = (HandLandmark.INDEX_FINGER_PIP,
                           HandLandmark.INDEX_FINGER_DIP,
                           HandLandmark.INDEX_FINGER_TIP)

_MIDDLE_FINGER_LANDMARKS = (HandLandmark.MIDDLE_FINGER_PIP,
                            HandLandmark.MIDDLE_FINGER_DIP,
                            HandLandmark.MIDDLE_FINGER_TIP)

_RING_FINGER_LANDMARKS = (HandLandmark.RING_FINGER_PIP,
                          HandLandmark.RING_FINGER_DIP,
                          HandLandmark.RING_FINGER_TIP)

_PINKY_FINGER_LANDMARKS = (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP,
                           HandLandmark.PINKY_TIP)

# Hands
_HAND_LANDMARK_STYLE = {
    _PALM_LANMARKS: DrawingSpec(color=_RED, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS: DrawingSpec(color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS: DrawingSpec(color=_PURPLE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS: DrawingSpec(color=_YELLOW, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS: DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS: DrawingSpec(color=_BLUE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}

hand_landmark_style = {}
for k, v in _HAND_LANDMARK_STYLE.items():
  for landmark in k:
    hand_landmark_style[landmark] = v
TASK_DIR = f'./data/MultiColor_skeleton_totalFrames'

#TODO: change task name in ['ori_bbox', 'skeletonCrop_bbox', 'default_skeleton', 'multiColor_skeleton', 'SegMask']
dataset_tpye = ['train', 'val', 'test']
SRC_VIDEO_DIR = f'../../Datasets/WLASL/videos_WLASL100/WLASL_100'
assert os.path.exists(str(SRC_VIDEO_DIR))
for d_type in dataset_tpye:
  src_videos = os.listdir(f'{SRC_VIDEO_DIR}/{d_type}')
  with tqdm(total=len(src_videos)) as pdbar:
    for video in sorted(src_videos):
      #video input
      video_name = f'{SRC_VIDEO_DIR}/{d_type}/{video}'
      assert os.path.exists(video_name)
      cap = cv2.VideoCapture(video_name)
      WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      
      #TODO: To visualize bbox on original video
      video_detectPerformancePath = f'{TASK_DIR}/WLASL_100/{d_type}/vis_detectedPerformance/'
      os.makedirs(video_detectPerformancePath, exist_ok=True)
      video_detectPerformance = cv2.VideoWriter(f'{video_detectPerformancePath}/detected_{video}', fourcc, 30, (WIDTH, HEIGHT))
      
      # print(f'save mask effect visualized video to: {video_detectPerformancePath}')
      
      video = video.replace('.mp4','')
      image_idx = 0
      with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        image_count = 0
        while cap.isOpened():
          image_count+=1
          success, image = cap.read()
          if not success:
            break

          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = hands.process(image)
          
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          if results.multi_hand_landmarks:
            hand_landmark_image = image.copy()
            hand_landmark_mask = np.zeros(image.shape[:], dtype=np.uint8)
            for hand_landmarks in results.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                  hand_landmark_image, hand_landmarks,
                  mp_hands.HAND_CONNECTIONS,
                  hand_landmark_style,
                  mp_drawing_styles.get_default_hand_connections_style())
              mp_drawing.draw_landmarks(
                  hand_landmark_mask, hand_landmarks, 
                  mp_hands.HAND_CONNECTIONS, 
                  hand_landmark_style,
                  mp_drawing_styles.get_default_hand_connections_style())
            
            # TODO: save landmark point and lines as mask
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/frameSkeleton/{video}', exist_ok=True)
            cv2.imwrite(f'{TASK_DIR}/WLASL_100/{d_type}/frameSkeleton/{video}/{image_idx:05}.png', hand_landmark_mask)            

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            visualize = np.zeros(image.shape)
            handlandmark_points = []
            
            for hand_landmark in results.multi_hand_landmarks:
              for idx, landmark in enumerate(hand_landmark.landmark):
                landmark_point = {}
                landmark_point['x'] = landmark.x * WIDTH
                landmark_point['y'] = landmark.y * HEIGHT
                landmark_point['label'] = idx
                handlandmark_points.append(landmark_point)
              x = [landmark.x *WIDTH for landmark in hand_landmark.landmark]
              y = [landmark.y *HEIGHT for landmark in hand_landmark.landmark]
              center = np.array([np.mean(x), np.mean(y)]).astype('int32')
              x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
              Bbox_bias = 0.03
              x0, y0, x1, y1 = int(x_min*(1-Bbox_bias)), int(y_min*(1-Bbox_bias)), int(x_max*(1+Bbox_bias)), int(y_max*(1+Bbox_bias))
              visualize[y0:y1, x0:x1, 0] = image[y0:y1, x0:x1, 0]
              visualize[y0:y1, x0:x1, 1] = image[y0:y1, x0:x1, 1]
              visualize[y0:y1, x0:x1, 2] = image[y0:y1, x0:x1, 2]
              cv2.rectangle(image, (x0, y0), (x1, y1), (255,0,0), 1)
              #TODO: MASK VALUE
              mask[y0:y1, x0:x1] = 5
            #TODO: landmark points
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/label_landmark/{video}', exist_ok=True)
            with open(f'{TASK_DIR}/WLASL_100/{d_type}/label_landmark/{video}/{image_idx:05}.json', 'w') as json_landmark:
              json.dump(handlandmark_points, json_landmark, indent=4)
            #TODO: TRAINING FRAME
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/frameOri/{video}', exist_ok=True)
            cv2.imwrite(f'{TASK_DIR}/WLASL_100/{d_type}/frameOri/{video}/{image_idx:05}.png', image)  
            #TODO: TRAINING MASK
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/mask/{video}', exist_ok=True)
            cv2.imwrite(f'{TASK_DIR}/WLASL_100/{d_type}/mask/{video}/{image_idx:05}.png', mask)
            # TODO: MASK IMAGE
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/visualize/{video}', exist_ok=True)
            cv2.imwrite(f'{TASK_DIR}/WLASL_100/{d_type}/visualize/{video}/{image_idx:05}.png', hand_landmark_image)
            cv2.putText(hand_landmark_image, 'frame: {}'.format(image_idx), (0,50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            video_detectPerformance.write(hand_landmark_image)
            image_idx += 1
          else:
            noLandmark_image = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            not_detectFrame = np.zeros(image.shape[:3], dtype=np.uint8)
            # TODO: save landmark point and lines as mask
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/frameSkeleton/{video}', exist_ok=True)
            cv2.imwrite(f'{TASK_DIR}/WLASL_100/{d_type}/frameSkeleton/{video}/{image_idx:05}.png', not_detectFrame)
            cv2.putText(noLandmark_image, 'notDetected_frame: {}'.format(image_idx), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            #SAVE all image including frames that is not detected
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/frameOri/{video}', exist_ok=True)
            cv2.imwrite(f'{TASK_DIR}/WLASL_100/{d_type}/frameOri/{video}/{image_idx:05}.png', image)
            #TODO: TRAINING MASK
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/mask/{video}', exist_ok=True)
            cv2.imwrite(f'{TASK_DIR}/WLASL_100/{d_type}/mask/{video}/{image_idx:05}.png', mask)
            # TODO: MASK IMAGE
            os.makedirs(f'{TASK_DIR}/WLASL_100/{d_type}/visualize/{video}', exist_ok=True)
            cv2.imwrite(f'{TASK_DIR}/WLASL_100/{d_type}/visualize/{video}/{image_idx:05}.png', mask)
            video_detectPerformance.write(noLandmark_image)
            image_idx += 1
      cap.release()
      pdbar.update()