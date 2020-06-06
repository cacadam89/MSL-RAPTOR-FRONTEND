#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src/yolov3')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src/SiamMask')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src/SiamMask/models')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src/SiamMask/experiments')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src/SiamMask/experiments/siammask_sharp')
from image_segmentor import ImageSegmentor
import cv2
import numpy as np
import time
import pdb

im = cv2.imread('SiamMask/data/tennis/00000.jpg')
i_s = ImageSegmentor(im, detect_classes_ids=[0, 80])

N = 100

d_tm = 0
t_tm = 0
for i in range(N):
    t0 = time.time()
    det = i_s.detect(im)
    if det is False:
        raise RuntimeError("DETECT FAILED")

    i_s.reinit_tracker(det,im)
    d_tm += time.time() - t0

    t0 = time.time()
    trck = i_s.track(im)
    t_tm += time.time() - t0

print("ave detect time = {:.4f}".format(d_tm / N))
print("ave track time = {:.4f}".format(t_tm / N))

print(det)
print(trck)
