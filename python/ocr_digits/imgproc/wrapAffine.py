# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''
import cv2
import math
import numpy as np


def affineTrans(img, angle=0., scale=1.):
    rangle = np.deg2rad(angle)
    h, w = img.shape[0], img.shape[1]

    rot_mat = cv2.getRotationMatrix2D(
            (w*0.5, h*0.5),
            angle, scale)

    img = cv2.warpAffine(
            img, rot_mat,
            (w, h),
            flags=cv2.INTER_LANCZOS4)

    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)

    return img


def affineTrans_about_center(img,
                             angle=0., scale=1.):
    rangle = np.deg2rad(angle)
    h, w = img.shape[0], img.shape[1]

    nh = (abs(np.sin(rangle)*h) +\
            abs(np.cos(rangle)*w))*scale
    nw = (abs(np.cos(rangle)*h) +\
            abs(np.sin(rangle)*w))*scale

    rot_mat = cv2.getRotationMatrix2D(
            (nw*0.5, nh*0.5),
            angle, scale)

    rot_move = np.dot(
            rot_mat,
            np.array([(nw-w)*0.5, (nh-h)*0.5,0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]

    img = cv2.warpAffine(
        img, rot_mat,
        (int(math.ceil(nw)), int(math.ceil(nh))),
        flags=cv2.INTER_LANCZOS4)

    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    return img


if __name__ == '__main__':
    import sys
    img = cv2.imread(sys.argv[1], 0)
    h, w = img.shape[0], img.shape[1]
    size = h if h < w else w
    img = affineTrans(img, scale=28.0/size)
    cv2.imshow('test', img)
    cv2.waitKey(0)
