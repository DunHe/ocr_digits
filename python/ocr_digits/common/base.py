# -*- coding: utf-8 -*-
'''
Create on 2016-01-26 19:13:47

@author: huangzhenghua
'''

import os
import imghdr
import string
import cv2
import numpy as np


class Base:

    def __init__(self, is_debug=True, debug_dir='/tmp/base_debug',
                 digit_w=28, digit_h=28):
        self.data = np.array([], dtype=np.uint8)
        self.target = np.array([], dtype=np.uint8)
        self.daima_data = np.array([], dtype=np.uint8)
        self.daima_target = np.array([], dtype=np.uint8)
        self.haoma_data = np.array([], dtype=np.uint8)
        self.haoma_target = np.array([], dtype=np.uint8)
        self.digit_w = digit_w
        self.digit_h = digit_h
        self.is_debug = is_debug
        self.debug_dir = debug_dir


    def show_field(self, img,
                   binImg,
                   img_path,
                   digits,
                   field_name,
                   field_x, field_y,
                   field_w, field_h,
                   ):
        key_name = field_name
        field_img = img[int(field_y): int(field_y) + int(field_h),
            int(field_x): int(field_x) + int(field_w)]
        basename = os.path.basename(img_path)
        basename = basename.replace(".jpg", "")
        field_path = os.path.join(
                self.debug_dir,
                "%s_%s.jpg" % (basename, key_name))
        cv2.imwrite(field_path, field_img)
        bin_field_path = os.path.join(
                self.debug_dir,
                "%s_%s_bin.jpg" % (basename, key_name))
        cv2.imwrite(bin_field_path, binImg)
        i = 0
        digit_shape = (self.digit_w, self.digit_h)
        for digit in digits:
            digit_path = os.path.join(
                self.debug_dir,
                "%s_%s_%d.jpg" % (basename, key_name, i))
            cv2.imwrite(digit_path, digit.reshape(digit_shape))
            i += 1

    def traverseFolders(self, action, dir='', ftype=''):
        curdir = os.path.realpath(dir)
        
        if os.path.isfile(curdir):
            if ftype is 'image':
                if imghdr.what(curdir):
                    action(curdir)
            if ftype is 'ttf':
                if curdir[-4:] == '.ttf':
                    action(curdir)

        elif os.path.isdir(curdir):
            for dir in os.listdir(curdir):
                if dir[0] != '.':
                    self.traverseFolders(
                        action, 
                        dir=os.path.join(curdir, dir),
                        ftype=ftype)
