# -*- coding: utf-8 -*-
'''
Create on 2016-01-19 16:16:16

@author: huangzhenghua
'''

import os
import imghdr
import string


def traverseFolders(action, dir='', ftype=''):
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
                traverseFolders(
                    action, 
                    dir=os.path.join(curdir, dir),
                    ftype=ftype)
