import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import colorsys

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox
from torchvision import transforms

def get_boxes(im0s, model, imgsz, stride, device):
    start_prepare = time.time()

    # Initialize
    half = device.type != 'cpu'  # half precision only supported on CUDA
    img = im0s

    img = img.half() if half else img.float()  # uint8 to fp16/32

    # Inference
    t1 = time.time()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.50, 0.55)
    t2 = time.time()
    # print(f"yolo time cost = {t2-t1}")

    boxes = []
    scores = []
    labels = []

    det = pred[0]
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape[2:]).round()
        # Record results
        for *xyxy, score, cls in reversed(det):
            boxes.append([x.item() for x in xyxy])
            scores.append(score.item())
            labels.append(int(cls))

    return boxes, scores, labels
