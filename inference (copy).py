"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort
import cv2

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime, vis_frame
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter
from alphapose.utils.transforms import heatmap_to_coord_simple
from test import drawer
from filterpy.kalman import KalmanFilter, update, predict, batch_filter
from filterpy.common import Q_discrete_white_noise, kinematic_kf, Saver
from scipy.linalg import block_diag, norm
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

"""----------------------------- options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
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

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def check_input():
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # for detection results
    if len(args.detfile):
        if os.path.isfile(args.detfile):
            detfile = args.detfile
            return 'detfile', detfile
        else:
            raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]

        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1

def draw_img(orig_img, preds):
    orig_img = cv2.line(orig_img, (preds[0][0], preds[0][1]), (preds[1][0], preds[1][1]), red1, 1)
    orig_img = cv2.line(orig_img, (preds[1][0], preds[1][1]), (preds[3][0], preds[3][1]), red1, 1)
    orig_img = cv2.line(orig_img, (preds[0][0], preds[0][1]), (preds[2][0], preds[2][1]), red2, 1)
    orig_img = cv2.line(orig_img, (preds[2][0], preds[2][1]), (preds[4][0], preds[4][1]), red2, 1)

    orig_img = cv2.line(orig_img, (preds[5][0], preds[5][1]), (preds[6][0], preds[6][1]), (255, 50, 50), 2)
    orig_img = cv2.line(orig_img, (preds[6][0], preds[6][1]), (preds[8][0], preds[8][1]), blue1, 2)
    orig_img = cv2.line(orig_img, (preds[5][0], preds[5][1]), (preds[7][0], preds[7][1]), blue2, 2)
    orig_img = cv2.line(orig_img, (preds[8][0], preds[8][1]), (preds[10][0], preds[10][1]), blue1, 2)
    orig_img = cv2.line(orig_img, (preds[7][0], preds[7][1]), (preds[9][0], preds[9][1]), blue2, 2)

    orig_img = cv2.line(orig_img, (preds[5][0], preds[5][1]), (preds[11][0], preds[11][1]), green1, 2)
    orig_img = cv2.line(orig_img, (preds[6][0], preds[6][1]), (preds[12][0], preds[12][1]), green2, 2)
    orig_img = cv2.line(orig_img, (preds[11][0], preds[11][1]), (preds[13][0], preds[13][1]), green1, 2)
    orig_img = cv2.line(orig_img, (preds[12][0], preds[12][1]), (preds[14][0], preds[14][1]), green2, 2)
    
    #cv2.putText(orig_img, f'{float(score[0]):.2f}', (70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    return orig_img

def kalman(t_1, dt, x_ndim=1, P_diag=(1., 1, 1, 1), R_std=1.,
                        Q_var=0.0001):
    """ helper, constructs 1d, constant velocity filter"""
    # f = KalmanFilter(dim_x=2, dim_z=2)
    # f.F = np.array([[1., 0.],
    #                 [0., 1.]])
    # f.H = np.array([[1., 0.],
    #                 [0., 1.]])
    # f.P = np.array([[1000., 0.],
    #                 [0., 1000.]])
    # f.R = np.array([[5., 0.],
    #                 [0., 5.]])
    # f.x = np.array([[128., 96.]])
    # kf = KalmanFilter(dim_x=2, dim_z=2)

    # kf.x = np.array([[t_2[0], t_2[1], t_1[0], t_1[1]]]).T
    # kf.P *= np.diag(P_diag)
    # kf.F = np.array([[1., dt, 0., 0.],
    #                  [0., 1., 0., 0.],
    #                  [0., 0., 1., dt],
    #                  [0., 0., 0., 1.]])

    # kf.H = np.array([[1., 0., 1., 0.],
    #                  [1., 0., 1., 0.]])

    # kf.R *= np.eye(2) * (R_std**2)
    # q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)
    # kf.Q = block_diag(q, q)

    kf = KalmanFilter(dim_x=2, dim_z=2)

    #kf.x = np.array([[t_1[0], t_1[1]]]).T
    kf.x = np.array([[1., 0.]]).T
    #kf.P = np.eye(2) * 1000.
    kf.P = np.array([[186, 0],
                    [0, 610.]])
    kf.F = np.array([[1, 0.],
                    [0, 1.]])

    kf.H = np.array([[1., 0.],
                     [0., 1.]])

    #kf.R *= np.eye(2) * (R_std**2)
    kf.R = np.array([[186, 0],
                    [0, 610]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=256)

    return kf

if __name__ == "__main__":
    mode, input_source = check_input()

    red1 = (100, 200, 255)
    red2 = (200, 100, 255)
    blue1 = (255, 100, 0)
    blue2 = (255, 0, 100)
    green1 = (150, 255, 50)
    green2 = (50, 255, 150)

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    if mode == 'webcam':
        det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
        det_worker = det_loader.start()
    elif mode == 'detfile':
        det_loader = FileDetectionLoader(input_source, cfg, args)
        det_worker = det_loader.start()
    else:
        det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

    pose_model.to(args.device)
    pose_model.eval()
    image_drawer = drawer(0)

    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    cv2.imshow("Demo", orig_img)
                    continue
                inps = inps.to(args.device)
                hm = pose_model(inps)
                hm = hm.cpu()
                preds, _ = heatmap_to_coord_simple(hm[0], cropped_boxes[0])
                t_1 = np.zeros((17, 2))
                t_2 = np.zeros((17, 2))
                if i == 0:
                    t_2 = preds
                elif i == 1:
                    t_1 = preds
                    f0 = kalman(t_1=t_1[0], dt=1.0, R_std=1., Q_var=5.1)
                    f1 = kalman(t_1=t_1[1], dt=1.0, R_std=1., Q_var=5.1)
                    f2 = kalman(t_1=t_1[2], dt=1.0, R_std=1., Q_var=5.1)
                    f3 = kalman(t_1=t_1[3], dt=1.0, R_std=1., Q_var=5.1)
                    f4 = kalman(t_1=t_1[4], dt=1.0, R_std=1., Q_var=5.1)
                    f5 = kalman(t_1=t_1[5], dt=1.0, R_std=1., Q_var=5.1)
                    f6 = kalman(t_1=t_1[6], dt=1.0, R_std=1., Q_var=5.1)
                    f7 = kalman(t_1=t_1[7], dt=1.0, R_std=1., Q_var=5.1)
                    f8 = kalman(t_1=t_1[8], dt=1.0, R_std=1., Q_var=5.1)
                    f9 = kalman(t_1=t_1[9], dt=1.0, R_std=1., Q_var=5.1)
                    f10 = kalman(t_1=t_1[10], dt=1.0, R_std=1., Q_var=5.1)
                    f11 = kalman(t_1=t_1[11], dt=1.0, R_std=1., Q_var=5.1)
                    f12 = kalman(t_1=t_1[12], dt=1.0, R_std=1., Q_var=5.1)
                    f13 = kalman(t_1=t_1[13], dt=1.0, R_std=1., Q_var=5.1)
                    f14 = kalman(t_1=t_1[14], dt=1.0, R_std=1., Q_var=5.1)
                else:
                    z = np.array(preds[0]).reshape(2, 1)
                    f0.update(z)
                    f0.predict()
                    preds[0] = z.reshape(1, 2)
                    z = np.array(preds[1]).reshape(2, 1)
                    f1.update(z)
                    f1.predict()
                    preds[1] = z.reshape(1, 2)
                    z = np.array(preds[2]).reshape(2, 1)
                    f2.update(z)
                    f2.predict()
                    preds[2] = z.reshape(1, 2)
                    z = np.array(preds[3]).reshape(2, 1)
                    f3.update(z)
                    f3.predict()
                    preds[3] = z.reshape(1, 2)
                    z = np.array(preds[4]).reshape(2, 1)
                    f4.update(z)
                    f4.predict()
                    preds[4] = z.reshape(1, 2)
                    z = np.array(preds[5]).reshape(2, 1)
                    f5.update(z)
                    f5.predict()
                    preds[5] = z.reshape(1, 2)
                    z = np.array(preds[6]).reshape(2, 1)
                    f6.update(z)
                    f6.predict()
                    preds[6] = z.reshape(1, 2)
                    z = np.array(preds[7]).reshape(2, 1)
                    f7.update(z)
                    f7.predict()
                    preds[7] = z.reshape(1, 2)
                    z = np.array(preds[8]).reshape(2, 1)
                    f8.update(z)
                    f8.predict()
                    preds[8] = z.reshape(1, 2)
                    z = np.array(preds[9]).reshape(2, 1)
                    print('----------------------')
                    print(z)
                    f9.predict()
                    f9.update(z)
                    preds[9] = f9.x[0, 0]
                    print(f9.x)
                    print(f9.K)
                    print('----------------------')
                    z = np.array(preds[10]).reshape(2, 1)
                    f10.update(z)
                    f10.predict()
                    preds[10] = z.reshape(1, 2)
                    z = np.array(preds[11]).reshape(2, 1)
                    f11.update(z)
                    f11.predict()
                    preds[11] = z.reshape(1, 2)
                    z = np.array(preds[12]).reshape(2, 1)
                    f12.update(z)
                    f12.predict()
                    preds[12] = z.reshape(1, 2)
                    z = np.array(preds[13]).reshape(2, 1)
                    f13.update(z)
                    f13.predict()
                    preds[13] = z.reshape(1, 2)
                    z = np.array(preds[14]).reshape(2, 1)
                    f14.update(z)
                    f14.predict()
                    preds[14] = z.reshape(1, 2)
            img = draw_img(orig_img, preds)
            image_drawer.show(img)
        print_finish_info()
        det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        if args.sp:
            det_loader.terminate()
        else:
            det_loader.terminate()
            det_loader.clear_queues()

