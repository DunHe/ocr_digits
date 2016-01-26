# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''
import cv2
import numpy as np


def bitReverse(img, thresh=True):
    if thresh:
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

    ## count white and black pixel of four edges
    white_cnt, black_cnt = 0, 0
    h, w = np.shape(img)[0], np.shape(img)[1]
    for i in range(h):
        white_cnt, black_cnt =\
                white_cnt + [0, 1][img[i][0] == 255],\
                black_cnt + [1, 0][img[i][0] == 255]
        white_cnt, black_cnt =\
                white_cnt + [0, 1][img[i][w-1] == 255],\
                black_cnt + [1, 0][img[i][w-1] == 255]

    for i in range(w):
        white_cnt, black_cnt =\
                white_cnt + [0, 1][img[0][i] == 255],\
                black_cnt + [1, 0][img[0][i] == 255]
        white_cnt, black_cnt =\
                white_cnt + [0, 1][img[h-1][i] == 255],\
                black_cnt + [1, 0][img[h-1][i] == 255]

    img = 255 - img if white_cnt > black_cnt else img

    return img
