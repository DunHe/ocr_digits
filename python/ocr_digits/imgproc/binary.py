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

    digit = cv2.resize(digit, (w, h))

    for i in range(h):
        for j in range(w):
            if digit[i][j] != 0:
                digit[i][j] = 255
    return digit


# def histogram_maxarea(hist):
    # maxarea = -1
    # start, left, right = 0, 0, 0
    # while start < hist.shape[0]:
        # area = 0
        # if hist[start] != 0:
            # localMax = hist[start]
            # tmp = start
            # while start < hist.shape[0] and hist[start] != 0:
                # if np.abs(tmp-start) > 50 and np.abs(hist[start] - localMax) < localMax*0.5:
                    # break
                # if localMax < hist[start]:
                    # localMax = hist[start]
                # area += hist[start]
                # start += 1
            # if area > maxarea:
                # maxarea = area
                # left = tmp
                # right = start
        # start += 1
    # return left, right


# def histogram_maxarea(hist):
    # maxIndex = hist.argmax()
    # left = maxIndex - 100 if maxIndex - 100 >= 0 else 0
    # right = maxIndex + 100 if maxIndex + 100 <= 255 else 255
    # return left, right

# def histogram_process(img):
    # h, w = img.shape[0], img.shape[1]
    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # left, right = histogram_maxarea(hist)

    # for i in range(h):
        # for j in range(w):
            # if img[i][j] < left or img[i][j] > right:
                # img[i][j] = (left+right)/2
    # return img
