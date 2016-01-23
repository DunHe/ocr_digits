# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''
import cv2
import numpy as np


def findContours(imgray,
                 min_area=25, max_area=2500,
                 min_h=5, min_w=5):

    imgray = cv2.adaptiveThreshold(
        imgray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)

    _, contours, _ = cv2.findContours(
        imgray.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    pst = [[(0, 0), (0, 0)]]

    filtered_contours = []

    for cnt in contours:
        flag = True
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h > min_area and h > min_h and\
           w > min_w and w * h < max_area:

            for i in pst:
                if(i[0][0] < x and i[0][1] < y and\
                   i[1][0] > x+w and i[1][1] > y+h):
                    flag = False
                    break
            if flag:
                pst.append([(x, y), (x+w, y+h)])
                filtered_contours.append(cnt)

    contours = filtered_contours

    return contours


def drawContours(imgray, contours):

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(imgray,
                      (x, y), (x + w, y + h),
                      (np.random.randint(0, 255, (3))), 1)
