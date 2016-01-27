# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''
import cv2
import numpy as np


def bitReverse(img):
    white_cnt, black_cnt = 0, 0
    h, w = img.shape[0], img.shape[1]
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

    return 255 - img if white_cnt > black_cnt else img


def digit_resize(digit, h, w):

    digit = cv2.resize(digit, (h, w))

    for i in range(h):
        for j in range(w):
            if digit[i][j] != 0:
                digit[i][j] = 255
    return digit
