# -*- coding: utf-8 -*-
'''
Create on 2016-01-27 21:20:18

@author: huangzhenghua
'''

import cv2
import numpy as np


def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)
