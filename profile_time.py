import cv2
import os
import shutil
import time
import json

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# (height, width), sorted in increasing order of the height
waymo_image_sizes = [
    (64, 96),
    (128, 192),
    (256, 384),
    (480, 640),
    (640, 960),
    (640, 1280),
    # the following mainly for merged scheduler
    (640, 64),
    (640, 1920),
    (960, 96),
    (960, 1920),
    (1280, 128),
    (1280, 160),
    (1280, 1920)
]

kitti_image_sizes = [
    (64, 96),
    (128, 192),
    (256, 192),
    (256, 384),
    (256, 512),
    # the following mainly for merged scheduler
    (384, 512),
    (384, 768),
    (384, 1280),
]


def read_time_profile(time_profile_file):
    '''
    Load and read the time profile file, convert the keys to int.
    '''
    with open(time_profile_file, 'r') as f:
        raw_time_profile = json.load(f)

    time_profile = {}

    for w in raw_time_profile:
        int_w = int(w)
        time_profile[int_w] = {}
        for h in raw_time_profile[w]:
            int_h = int(h)
            time_profile[int_w][int_h] = {}
            for b in raw_time_profile[w][h]:
                int_b = int(b)
                time_profile[int_w][int_h][int_b] = raw_time_profile[w][h][b]

    return time_profile


def profile_one_image_size(model, w, h, b, test_count, half, device=0):
    '''
    Use this function to profile a given image size.
    '''
    image_size = [3, h, w]
    print("Profiling image size: ", image_size, ", batch size: ", b)

    # generate random data and warmup the model, data input is between [0, 1]
    image_path = '/home/sl29/data/Waymo/validation_images/segment-2094681306939952000/1507658568795604-FRONT.jpeg'
    random_data = cv2.imread(image_path)

    # convert color from BGR to RGB
    random_data = cv2.cvtColor(random_data, cv2.COLOR_BGR2RGB)

    # resize to (h, w)
    random_data = cv2.resize(random_data, (h, w))

    # normalize
    random_data = np.array(random_data) / 255

    # from [h, w, color] to [color, h, w]
    random_data = random_data.transpose(2, 0, 1)  # color, h, w

    # add batch dimension
    random_data = random_data[np.newaxis, ...].astype(np.float32)
    random_data = np.tile(random_data, [b, 1, 1, 1])

    # convert to torch tensor
    random_tensor = torch.from_numpy(random_data).to(device)

    #  from float 32 to float 16
    random_tensor = random_tensor.half() if half else random_tensor.float()

    # warmup run
    for _ in range(3):
        # inference
        # return [
        #   1) concated output bboxes, [batch_size, #bbox, 80(coco classes)+5],
        #   2) list of output at each output layer with shape [batch_size, #anchor, h, w, 80+5]
        pred = model(random_tensor)[0]

        # nms, pred: [batch_size, #bbox, 80(coco classes)+5]
        pred = non_max_suppression(pred)

    prediction_time_list = []
    post_processing_time_list = []

    for _ in range(test_count):
        # inference
        start = time_synchronized()

        pred = model(random_tensor)[0]

        end = time_synchronized()
        prediction_time_list.append(end - start)

        # postprocess by NMS
        start = time_synchronized()
        pred = non_max_suppression(pred)
        end = time_synchronized()
        post_processing_time_list.append(end - start)

    mean_inference_time = np.mean(prediction_time_list)
    mean_postprocess_time = np.mean(post_processing_time_list)
    print(f'mean inference time: {mean_inference_time} s, mean_postprocess_time: {mean_postprocess_time} s')

    return mean_inference_time, mean_postprocess_time


if __name__ == '__main__':
    profile_path = '/home/sl29/DeepScheduling/result/yolov5_result/yolov5_profiles'
    machine_name = str(np.loadtxt(os.path.join(profile_path, 'machine.txt'), dtype=str))
    dataset = 'waymo'
    model = "yolov5s"
    profile_file = os.path.join(profile_path, model + '_' + dataset + '_' + machine_name + '.json')
    weight_file = "weights/" + model + ".pt"

    # -------------------------------- Model Initialization --------------------------------
    # set device as the first GPU by default
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weight_file, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    # the model is FP16 for GPU
    if half:
        model.half()

    # -------------------------------- End of Initialization --------------------------------

    # perform (incremental) time profiling
    if not os.path.exists(profile_file):
        image_size_times = dict()
    else:
        image_size_times = read_time_profile(profile_file)

    start = time.time()

    if dataset == 'kitti':
        image_sizes = kitti_image_sizes
    else:
        image_sizes = waymo_image_sizes

    for (h, w) in image_sizes:
        if w not in image_size_times:
            image_size_times[w] = {}

        if h not in image_size_times[w]:
            image_size_times[w][h] = {}

        # decide batch limit
        # batch_limit = max(1920 // w + 3, 1280 // h + 3)
        # batch_limit = max(batch_limit, 6)
        # batch_limit = min(batch_limit, 10)
        batch_limit = 11

        for b in range(1, batch_limit):
            if w in image_size_times:
                if h in image_size_times[w]:
                    if b in image_size_times[w][h]:
                        continue

            mean_inference_time, mean_postprocess_time = profile_one_image_size(model, w, h, b,
                                                                                200, half, device)
            image_size_times[w][h][b] = {
                'inference': mean_inference_time,
                'postprocess': mean_postprocess_time
            }

    # save the file
    with open(profile_file, 'w') as f:
        f.write(json.dumps(image_size_times, indent=4))

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
