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

    cropImg = cv2.adaptiveThreshold(
        cropImg,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)

    cropImg = Xihua(cropImg, thinArray)

    cropImg = bitReverse(cropImg)

    cv2.imshow('test', cropImg)
    cv2.waitKey(0)

    min_start_x, min_char_width, min_col_width = search_char_x(
            formatDigit(cropImg),
            num_chars)
    digits = []

    for i in range(num_chars):
        left = min_start_x + (min_char_width + min_col_width) * i
        _digit = cropImage(cropImg, left, 0, img_h, min_char_width)
        _digit = resizeDigit(_digit, 100, 100)

        # _digit = cv2.morphologyEx(_digit, cv2.MORPH_OPEN, kernel)
        # _digit = cv2.morphologyEx(_digit, cv2.MORPH_CLOSE, kernel)
        _digit = resizeDigit(_digit, digit_w, digit_h, thresh='manual', bitwise=False)
        _digit = formatDigit(_digit, 1, 784, thresh=False, bitwise=False)

        digits.append(_digit)
    return digits
