# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''

import cv2
import numpy as np
from ocr_digits.ml.format import formatDigit
from ocr_digits.ml.format import resizeDigit
from ocr_digits.imgproc.wrapAffine import affineTrans
from ocr_digits.imgproc.binaryProc import bitReverse
from ocr_digits.imgproc.char_segments import search_char_x
from ocr_digits.imgproc.edgeDetect import CLAHE
from ocr_digits.imgproc.edgeDetect import Sobel
from ocr_digits.reco_digits.contours import findContours
from ocr_digits.reco_digits.contours import drawContours
from ocr_digits.imgproc.binaryProc import thinArray
from ocr_digits.imgproc.binaryProc import Xihua 
import copy


def cropImage(img, img_x, img_y, img_h, img_w):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img[int(img_y) : int(img_y) + int(img_h),
            int(img_x) : int(img_x) + int(img_w)]


def cropDigits(img, img_x, img_y, img_h, img_w, num_chars, digit_w=28, digit_h=28):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cropImg = cropImage(img, img_x, img_y, img_h, img_w)

    cropImg = cv2.GaussianBlur(cropImg, (5, 5), 0)

    # http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html#gsc.tab=0
    ret3, binImg = cv2.threshold(cropImg, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#    cropImg = cv2.adaptiveThreshold(
#        cropImg,
#        255,
#        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#        cv2.THRESH_BINARY, 11, 2)

#    cv2.imshow('test', binImg)
#    cv2.waitKey(0)

    #cropImg = Xihua(cropImg, thinArray)


#    cv2.imshow('test', binImg)
#    cv2.waitKey(0)

    binImg = bitReverse(binImg)

    if cv2.__version__[0] == "2":
        contours, hierarchy = cv2.findContours(
            copy.deepcopy(binImg),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(
            copy.deepcopy(binImg),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

    
#    cv2.imshow('test', binImg)
#    cv2.waitKey(0)

    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 20 and w > 5 and h > 5:
            rects.append((w*h, (x, y, w, h)))

    # sorted by area, get the max area rects
    sorted_rects = sorted(rects, key=lambda x: x[0], reverse=True)
    if len(sorted_rects) > num_chars:
        digit_rects = sorted_rects[:num_chars]
    else:
        digit_rects = sorted_rects

    # sorted by x from left to right
    x_rects = []
    for digit_rect in digit_rects:
        x = digit_rect[1][0]
        rect = digit_rect[1]
        x_rects.append((x, rect))

    left_to_right_rects = sorted(x_rects, key=lambda x: x[0])

    digits = []
    for i in range(num_chars):
        if i < len(left_to_right_rects):
            rect = left_to_right_rects[i][1]
            _digit = cropImage(binImg, rect[0], rect[1], rect[3],
                               rect[2])
            _digit = resizeDigit(_digit,
                             h=digit_w, w=digit_h,
                             thresh='manual', bitwise=False)
            _digit = formatDigit(_digit,
                             h=1, w=digit_w*digit_h,
                             thresh=False, bitwise=False)
            digits.append(_digit)
        else:
            _digit = np.zeros((digit_w*digit_h))
            digits.append(_digit)

##    cv2.imshow('test', cropImg)
##    cv2.waitKey(0)
#
#    min_start_x, min_char_width, min_col_width = search_char_x(
#            formatDigit(binImg),
#            num_chars)
#    digits = []
#
#    for i in range(num_chars):
#        left = min_start_x + (min_char_width + min_col_width) * i
#        _digit = cropImage(binImg, left, 0, img_h, min_char_width)
#
#        _digit = resizeDigit(_digit,
#                             h=digit_w, w=digit_h,
#                             thresh='manual', bitwise=False)
#        _digit = formatDigit(_digit,
#                             h=1, w=784,
#                             thresh=False, bitwise=False)
#
#        digits.append(_digit)
    return digits, binImg
