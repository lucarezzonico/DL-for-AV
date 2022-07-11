
# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import mediapipe as mp

# detector imports
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

data = []


class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        super(Detector, self).__init__()

        self.FRAME_WIDTH = 160  # 160  # 640
        self.FRAME_HEIGHT = 120  # 120  # 480
        self.FRAME_HEIGHT_ROUNDED = 128  # 120  # 480
        

        self.mp_hands = mp.solutions.hands

        # Store the indexes of the tips landmarks of each finger of a hand in a list.
        self.fingers_tips_ids = [self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                 self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.PINKY_TIP]

        # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
        self.model_statuses = {'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False, 'LEFT_RING': False, 'LEFT_PINKY': False,
                               'RIGHT_THUMB': False, 'RIGHT_INDEX': True, 'RIGHT_MIDDLE': True, 'RIGHT_RING': True, 'RIGHT_PINKY': True}

        self.person_to_track = 0
        self.correct_symbol = False

        # TODO: MEAN & STD
        self.mean = [[[[0.5548078, 0.56693329, 0.53457436]]]]
        self.std = [[[[0.26367019, 0.26617227, 0.25692861]]]]
        # self.img_size = 100
        # self.img_size_w = 80
        # self.img_size_h = 60
        # self.min_object_size = 10
        # self.max_object_size = 40
        # self.num_objects = 1
        # self.num_channels = 3



        # webcam test initialization
        self.save_pic, self.imgsz, self.half = True, [self.FRAME_HEIGHT_ROUNDED], False
        self.imgsz *= 2 if len(self.imgsz) == 1 else 1  # expand

        self.yolo_model = 'yolov5x.pt'
        self.deep_sort_model = 'osnet_x0_25'

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(self.deep_sort_model, max_dist=cfg.DEEPSORT.MAX_DIST,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        # Initialize
        self.device = select_device('')
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.yolo_model, device=device, dnn=True)
        stride, names, pt, jit, _ = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        self.half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            self.model.model.half() if self.half else self.model.model.float()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if pt and device.type != 'cpu':
            self.model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(self.model.model.parameters())))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0, 0.0], 0

    def countFingers(self, results, fingers_statuses):
        '''
        Params:
            image:            The image of the hands on which the fingers counting is required to be performed.
            hand_landmarks:   The output of the hands landmarks detection performed on the image of the hands.
            hand_label:       'Left' or 'Right'
            fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        Returns:
            fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        '''

        for hand_index, hand_info in enumerate(results.multi_handedness):

            hand_label = hand_info.classification[0].label

            hand_landmarks = results.multi_hand_landmarks[hand_index]

            # Iterate over the indexes of the tips landmarks of each finger of the hand.
            for tip_index in self.fingers_tips_ids:

                # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
                finger_name = tip_index.name.split("_")[0]

                # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
                if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                    # Update the status of the finger in the dictionary to true.
                    fingers_statuses[hand_label.upper() + "_" + finger_name] = True

            # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
            thumb_tip_x = hand_landmarks.landmark[4].x
            thumb_mcp_x = hand_landmarks.landmark[2].x

            # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
            if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (
                    hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
                # Update the status of the thumb in the dictionary to true.
                fingers_statuses[hand_label.upper() + "_THUMB"] = True

        # Return the status of each finger
        return fingers_statuses

    def process_img(self, im0, left, top, right, bottom, m_statuses):
        # create transparent overlay for bounding box
        # bbox_array = np.zeros([self.FRAME_HEIGHT,self.FRAME_WIDTH,4], dtype=np.uint8)

        cropped_img = im0.copy()

        cropped_img[:, 0:left, :] = 0
        cropped_img[:, right:self.FRAME_WIDTH, :] = 0
        cropped_img[0:top, :, :] = 0
        cropped_img[bottom:self.FRAME_HEIGHT_ROUNDED, :, :] = 0
        
        
        
        cropped_img = cv2.flip(cropped_img, 1)

        # Make Detections
        with self.mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1) as hands:
            results = hands.process(cropped_img)

        # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
        fingers_statuses = {'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False, 'LEFT_RING': False, 'LEFT_PINKY': False,
                            'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False, 'RIGHT_PINKY': False}
                            

        if results.multi_hand_landmarks:
            fingers_statuses = self.countFingers(results, fingers_statuses)
            
        print(fingers_statuses)

        return fingers_statuses

    def check_symbol(self, statuses, m_statuses):
        self.correct_symbol = True
        for hand_label in ['Right', 'Left']:
            for tip_index in self.fingers_tips_ids:
                # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
                finger_name = tip_index.name.split("_")[0]

                # Compare the status of the fingers.
                if statuses[hand_label.upper() + "_" + finger_name] != m_statuses[
                    hand_label.upper() + "_" + finger_name]:
                    self.correct_symbol = False

            # Check for thumb
            if statuses[hand_label.upper() + "_THUMB"] != m_statuses[hand_label.upper() + "_THUMB"]:
                self.correct_symbol = False
                

        return self.correct_symbol

    def forward(self, img):

        img = np.array(img) # RGB opencv image

        pred_y_label = [0]
        pred_bboxes = [0, 0, 0, 0]

        with torch.no_grad():

            img_original = torch.from_numpy(img).to(self.device)
            
            img = img_original
            
            
            black = torch.zeros(self.FRAME_HEIGHT_ROUNDED, self.FRAME_WIDTH, 3).to(self.device)
            edge = int((self.FRAME_HEIGHT_ROUNDED - self.FRAME_HEIGHT) / 2)
            black[edge : self.FRAME_HEIGHT + edge, 0:self.FRAME_WIDTH, :] = img_original
            
            img = black

            
            im0 = img.cpu().numpy().astype(np.uint8)
            
            img = img.permute(2,0,1)
            
            cv2.imwrite('photo_out.jpg', im0)
            
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            
            
            # img : float tensor (0-1) 1 x 3 x 480 x 640
            # im0 : uint8 numpy ndarray (0-255) 480 x 640 x 3

            # Inference + Apply NMS
            pred = self.model(img, augment=True, visualize=False) # img : float tensor (0-1) 1 x 3 x 480 x 640
            pred = non_max_suppression(prediction=pred, conf_thres=0.5, iou_thres=0.5, classes=0, agnostic=True, max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                self.seen += 1

                annotator = Annotator(im0, line_width=2, pil=not ascii)
                w, h, _ = im0.shape[1], im0.shape[0], im0.shape[2]

                # im0 = 480 x 640 x 3

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    


                    xywhs = xyxy2xywh(det[:, 0:4])

                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    # check which person has to be tracked
                    if len(outputs) > 0:
                        for output in outputs:
                            left, top, right, bottom = output[0], output[1], output[2], output[3]
                            id = output[4]
                            if self.person_to_track == 0:
                            
                                                               
                                fingers_statuses = self.process_img(im0, left, top, right, bottom, self.model_statuses)
                                # im0 = reference_frame[:,:,:]
                                self.correct_symbol = self.check_symbol(fingers_statuses, self.model_statuses)

                                # track the first person that shows the sign
                            if self.correct_symbol and self.person_to_track == 0:
                                print('track person ', id)
                                self.person_to_track = id

                            if id == self.person_to_track:
                                print('tracking person ', id)
                                pred_y_label = [1] # ["person_of_interest"]
                                pred_bboxes = [round(left + (right - left) / 2), round(top + (bottom - top) / 2 - edge),
                                               (right - left), (bottom - top)]
                            else:
                                pred_y_label = [0] # [None]
                                pred_bboxes = [int(self.FRAME_WIDTH/2), int(self.FRAME_HEIGHT/2), 0, 0] # [None, None, None, None]

                            self.correct_symbol = False

                    
                    LOGGER.info('Person detected')

                else:
                    self.deepsort.increment_ages()
                    LOGGER.info('No detections')

        return pred_bboxes, pred_y_label


if __name__ == '__main__':

    pil_image = Image.open("photo.jpg")

    img_detector = Detector()
    img_detector.forward(pil_image)




