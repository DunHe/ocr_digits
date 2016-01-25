# -*- coding: utf-8 -*-
'''
Create on 2016-01-20 15:04:15

@author: huangzhenghua
'''

import cv2
import numpy as np
from ocr_digits.imgproc.binaryProc import bitReverse


def formatDigit(digit,
                h='', w='',
                thresh = 'automatic',
                bitwise = True):

    if h is not '' and w is not '':
        digit = digit.reshape((h, w))

        if thresh is 'manual':
            for i in range(h):
                for j in range(w):
                    if digit[i][j] != 0:
                        digit[i][j] = 255

    if thresh is 'automatic':
        digit = cv2.adaptiveThreshold(
            digit,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

    if bitwise:
        digit = bitReverse(digit, False)

    return digit


def resizeDigit(digit,
                h='', w='',
                thresh = 'automatic',
                bitwise = True):

    if h is not '' and w is not '':
        digit = cv2.resize(digit, (h, w))

        if thresh is 'manual':
            for i in range(h):
                for j in range(w):
                    if digit[i][j] != 0:
                        digit[i][j] = 255

    if thresh is 'automatic':
        digit = cv2.adaptiveThreshold(
            digit,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

    if bitwise:
        digit = bitReverse(digit, False)

    return digit
