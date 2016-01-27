# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''

import copy
import cv2
import numpy as np
from ocr_digits.imgproc.binproc import digit_resize
from ocr_digits.imgproc.char_segments import search_char_x
from ocr_digits.imgproc.imgstren import gamma_correction
from ocr_digits.imgcrop.contours import inner_findContours
from ocr_digits.imgcrop.contours import draw_contour_rect


def cropImage(img, img_x, img_y, img_h, img_w):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img[int(img_y) : int(img_y) + int(img_h),
            int(img_x) : int(img_x) + int(img_w)]


def cropDigits(img,
               img_x, img_y, img_h, img_w,
               num_chars,
               digit_w=28, digit_h=28):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cropImg = cropImage(img, img_x, img_y, img_h, img_w)
    cropImg = gamma_correction(cropImg, 2)
    cropImg = cv2.GaussianBlur(cropImg, (5, 5), 0)

    binImg, contours = inner_findContours(cropImg,
                                  contour_filter=True,
                                  thresh='otsu')
    # tmp = copy.deepcopy(cropImg)
    # draw_contour_rect(tmp, contours)
    # cv2.namedWindow('test', 1000)
    # cv2.imshow('test', tmp)
    # cv2.waitKey(0)

    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 20 and w > 5 and h > 5 and h >= w:
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
            _digit = digit_resize(_digit, digit_w, digit_h)
            _digit = _digit.reshape((1, digit_h*digit_w))
            digits.append(_digit)
        else:
            _digit = np.zeros((digit_w*digit_h))
            digits.append(_digit)

    return digits, binImg
