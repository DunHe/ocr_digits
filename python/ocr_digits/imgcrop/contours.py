# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''

import copy
import cv2
import numpy as np
from ocr_digits.imgproc.binaryProc import bitReverse


def inner_findContours(img, 
                 contour_filter=False,
                 thresh='otsu',
                 min_area=25, max_area=2500,
                 min_h=5, min_w=5):

    if thresh == 'otsu':
        _, img = cv2.threshold(img, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh == 'adaptive':
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

    img = bitReverse(img)

    if cv2.__version__[0] == "2":
        contours, hierarchy = cv2.findContours(
            copy.deepcopy(img),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(
            copy.deepcopy(img),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

    if contour_filter:
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > min_area and h > min_h and w > min_w and w * h < max_area:
                filtered_contours.append(cnt)
        contours = filtered_contours
    return img, contours


def draw_contour_rect(img, contours):

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img,
                      (x, y), (x + w, y + h),
                      (np.random.randint(0, 255, (3))), 1)
