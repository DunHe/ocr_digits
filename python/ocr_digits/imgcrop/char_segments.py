# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:53:26 2016

@author: jinpeng
"""

import numpy as np
import os
import cv2


def get_x_sep_col_score(bw_img, start_x, n_char,
                        char_width, col_width, is_x=True):
    start_xs = []
    for i in xrange(n_char):
        start_xs.append(start_x)
        start_x += (char_width+col_width)
    if is_x:
        x_bw_img = np.sum(bw_img, axis=0)
    else:
        x_bw_img = np.sum(bw_img, axis=1)
    score = 0
    for i, x in enumerate(start_xs):
        #print("x_bw_img[%d:%d]=%d" % (x, x+col_width,
        #      np.sum(x_bw_img[x:x+col_width])))
        score -= np.sum(x_bw_img[x:x+char_width])
        if i == (len(start_xs) - 1):
            score += np.sum(x_bw_img[x+char_width:x+char_width+col_width])
    return score


def search_char_x(bw_img, n_char, is_x=True):
    if is_x:
        max_char_width = int(bw_img.shape[1] / n_char)
    else:
        max_char_width = int(bw_img.shape[0] / n_char)

    min_char_width = int(max_char_width/2)

    start_xs = []
    for i in range(max_char_width-1, -1, -1):
        start_xs.append(i)

    possible_char_widths = []
    for i in range(min_char_width, max_char_width):
        possible_char_widths.append(i)

    possible_col_widths = []
    for i in range(1, min_char_width):
        possible_col_widths.append(i)

    # we need min score
    min_score = np.sum(bw_img)
    min_start_x = 0
    min_char_width = 0
    min_col_width = 0

    for start_x in start_xs:
        for possible_char_width in possible_char_widths:
            for possible_col_width in possible_col_widths:
                total_width = start_x + \
                    (possible_char_width + possible_col_width) * (n_char - 1) \
                    + possible_char_width
                if is_x:
                    if total_width > bw_img.shape[1]:
                        continue
                else:
                    if total_width > bw_img.shape[0]:
                        continue
                score = get_x_sep_col_score(
                    bw_img,
                    start_x,
                    n_char,
                    possible_char_width,
                    possible_col_width,
                    is_x)
                if score < min_score:
                    min_score = score
                    min_start_x = start_x
                    min_char_width = possible_char_width
                    min_col_width = possible_col_width
    return min_start_x, min_char_width, min_col_width


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_image = os.path.join(cur_dir, "..", "..", "..",
                              "example", "data", "digit_line.png")
    print("test_image=", test_image)

    num_chars = 12
    img = cv2.imread(test_image, 0)

    height = img.shape[0]

    bw_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    bw_img = 255 - bw_img
    cv2.imwrite("/tmp/bw_img.jpg", bw_img)
    min_start_x, min_char_width, min_col_width = search_char_x(bw_img,
                                                               num_chars)
    for i in range(num_chars):
        left = min_start_x + (min_char_width + min_col_width) * i
        right1 = min_start_x + (min_char_width + min_col_width) * i + min_char_width
        right2 = min_start_x + (min_char_width + min_col_width) * (i + 1)
        top = 0
        low = height - 1
        cv2.line(img, (left, top), (left, low), (255, 0, 0), 1)
        cv2.line(img, (right1, top), (right1, low), (0, 255, 0), 1)
        cv2.line(img, (right2, top), (right2, low), (0, 0, 255), 1)
    cv2.imwrite("/tmp/char_seg.jpg", img)
