# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''
import cv2
import numpy as np


thinArray = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
             1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]


def bitReverse(img, thresh=True):
    if thresh:
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

    ## count white and black pixel of four edges
    white_cnt, black_cnt = 0, 0
    h, w = np.shape(img)[0], np.shape(img)[1]
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

    img = 255 - img if white_cnt > black_cnt else img

    return img

# http://www.cnblogs.com/xianglan/archive/2011/01/01/1923779.html
# http://www.cnblogs.com/xiaowuyi/archive/2012/09/10/2675286.html

def VThin(image,array):
    h, w = image.shape[0], image.shape[1]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i,j-1]+image[i,j]+image[i,j+1] if 0<j<w-1 else 1
                if image[i,j] == 0  and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image
    

def HThin(image,array):
    h, w = image.shape[0], image.shape[1]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i-1,j]+image[i,j]+image[i+1,j] if 0<i<h-1 else 1   
                if image[i,j] == 0 and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image
    

def Xihua(image,array,num=10):
    iXihua = image.copy()
    for i in range(num):
        VThin(iXihua,array)
        HThin(iXihua,array)
    return iXihua
