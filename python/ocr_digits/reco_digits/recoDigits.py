# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''

import cv2
import numpy as np
from ocr_digits.ml.format import formatDigit
from ocr_digits.ml.format import resizeDigit
from ocr_digits.imgproc.binaryProc import bitReverse
from ocr_digits.imgproc.char_segments import search_char_x
from ocr_digits.reco_digits.contours import findContours
from ocr_digits.reco_digits.contours import drawContours


def cropDigits(img, x, y, h, w, num_chars, digit_w=28, digit_h=28):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cropImg = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            cropImg[i][j] = img[y+i][x+j]

    min_start_x, min_char_width, min_col_width = search_char_x(
            formatDigit(cropImg),
            num_chars)
    digits = []
    kernel = np.ones((3, 3), np.uint8)

    for i in range(num_chars):
        _digit = np.zeros((h, min_char_width), np.uint8)

        left = min_start_x + (min_char_width + min_col_width) * i
        for i in range(h):
            for j in range(min_char_width):
                _digit[i][j] = cropImg[i][left+j]

        _digit = resizeDigit(_digit, 100, 100)
        _digit = cv2.morphologyEx(_digit, cv2.MORPH_OPEN, kernel)
        _digit = cv2.morphologyEx(_digit, cv2.MORPH_CLOSE, kernel)
        _digit = resizeDigit(_digit, digit_w, digit_h, thresh='manual', bitwise=False)
        _digit = formatDigit(_digit, 1, 784, thresh=False, bitwise=False)

        digits.append(_digit)
    return digits
