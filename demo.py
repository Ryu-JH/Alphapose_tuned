import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as f

import argparse
import os
import sys
import time

from darknet import Darknet
from alphapose.models import builder
from alphapose.opt import cfg
from alphapose.utils.transforms import heatmap_to_coord_simple
from alphapose.utils.config import update_config

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=False,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

if __name__ == "__main__":
    device = torch.device("cuda")
    print('Loading AlphaPose model.....')
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    model.load_state_dict(torch.load('/home/ryu/AlphaPose/pretrained_models/fast_res50_256x192.pth'))
    model.to(device)
    model.eval()

    print('Loading YOLO model.....')
    detector = Darknet('detector/yolo/cfg/yolov3-spp.cfg')
    print(detector)
    detector.load_weights('detector/yolo/data/yolov3-spp.weights')
    detector.to(device)
    detector.eval()
    #detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    cap = cv2.VideoCapture("examples/ir_side2.avi")

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    delay = round(1000 / fps)

    red1 = (100, 200, 255)
    red2 = (200, 100, 255)
    blue1 = (255, 100, 0)
    blue2 = (255, 0, 100)
    green1 = (150, 255, 50)
    green2 = (50, 255, 150)
    while True:
        success, img = cap.read()
        # detector_result = detector(img)
        # print(detector_result)
        inps = cv2.resize(img, dsize=(256, 192))
        ptime = time.time()
        inps = np.array(img)
        inps = inps/255
        inps = np.swapaxes(inps, 0, 1)
        inps = np.swapaxes(inps, 0, 2)
        inps = inps[np.newaxis, ...]
        inps = torch.Tensor(inps)
        inps = inps.to(device)
        img = torch.Tensor(img)
        img = img.to(device)
        with torch.no_grad():
            det = detector(inps, args)
            det = det.to('cpu')
            print(det, det.shape)
            pred = model(inps)
            pred = pred.to('cpu')
            print(pred.shape)
            # for i in range(15):
            preds, _ = heatmap_to_coord_simple(pred[0], (0, 0, 256, 192))
        
        img = cv2.line(img, (preds[0][0], preds[0][1]), (preds[1][0], preds[1][1]), red1, 1)
        img = cv2.line(img, (preds[1][0], preds[1][1]), (preds[3][0], preds[3][1]), red1, 1)
        img = cv2.line(img, (preds[0][0], preds[0][1]), (preds[2][0], preds[2][1]), red2, 1)
        img = cv2.line(img, (preds[2][0], preds[2][1]), (preds[4][0], preds[4][1]), red2, 1)

        img = cv2.line(img, (preds[5][0], preds[5][1]), (preds[6][0], preds[6][1]), (255, 50, 50), 2)
        img = cv2.line(img, (preds[6][0], preds[6][1]), (preds[8][0], preds[8][1]), blue1, 2)
        img = cv2.line(img, (preds[5][0], preds[5][1]), (preds[7][0], preds[7][1]), blue2, 2)
        img = cv2.line(img, (preds[8][0], preds[8][1]), (preds[10][0], preds[10][1]), blue1, 2)
        img = cv2.line(img, (preds[7][0], preds[7][1]), (preds[9][0], preds[9][1]), blue2, 2)

        img = cv2.line(img, (preds[5][0], preds[5][1]), (preds[11][0], preds[11][1]), green1, 2)
        img = cv2.line(img, (preds[6][0], preds[6][1]), (preds[12][0], preds[12][1]), green2, 2)
        img = cv2.line(img, (preds[11][0], preds[11][1]), (preds[13][0], preds[13][1]), green1, 2)
        img = cv2.line(img, (preds[12][0], preds[12][1]), (preds[14][0], preds[14][1]), green2, 2)

        ctime = time.time()
        fps = 1 / (ctime-ptime)

        cv2.putText(img, f'FPS:{fps:.1f}', (70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
        #img = cv2.resize(img, dsize=(w, h))
        cv2.imshow("Image", img)
        key = cv2.waitKey(20)
        if key & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break    

