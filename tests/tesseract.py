# -*- coding: utf-8 -*-
'''
Create on 2016-01-27 11:47:06

@author: huangzhenghua
'''

import os
import cv2
import pytesseract
from PIL import Image
from ocr_digits.common.base import Base
from ocr_digits.reco_digits.recoDigits import cropImage


class Pytesser(Base):


    def process(self, dir):
        img = cv2.imread(dir, 0)

        daima = self.jsonLoad(dir+'.desc.json',
                         'fapiaodaima', 'bounding_box')
        daima_x = int(float(daima['top_left'][0]))
        daima_y = int(float(daima['top_left'][1]))
        daima_w = int(float(daima['low_right'][0])) - daima_x
        daima_h = int(float(daima['low_right'][1])) - daima_y
        daima_text = self.jsonLoad(dir+'.desc.json',
                'fapiaodaima', 'text').encode('utf-8')
        self.digits_num += len(daima_text)

        daima_img = cropImage(img, int(daima_x), int(daima_y), int(daima_h), int(daima_w))
        cv2.imwrite('./tmp.png', daima_img)
        daima_predicted = pytesseract.image_to_string(Image.open('./tmp.png'))
        self.cal_digits_right(dir, daima_text, daima_predicted)
        os.remove('./tmp.png')

        haoma = self.jsonLoad(dir+'.desc.json', 'fapiaohaoma', 'bounding_box')
        haoma_x = int(float(haoma['top_left'][0]))
        haoma_y = int(float(haoma['top_left'][1]))
        haoma_w = int(float(haoma['low_right'][0])) - haoma_x
        haoma_h = int(float(haoma['low_right'][1])) - haoma_y
        haoma_text = self.jsonLoad(dir+'.desc.json',
                'fapiaohaoma', 'text').encode('utf-8')
        self.digits_num += len(haoma_text)
        
        haoma_img = cropImage(img, int(haoma_x), int(haoma_y), int(haoma_h), int(haoma_w))
        cv2.imwrite('./tmp.png', haoma_img)
        haoma_predicted = pytesseract.image_to_string(Image.open('./tmp.png'))
        self.cal_digits_right(dir, haoma_text, haoma_predicted)
        os.remove('./tmp.png')

    
    def cal_digits_right(self, dir, expected, predicted):
        size = len(expected) if len(expected) < len(predicted) else len(predicted)

        for i in range(size):
            if expected[i] == predicted[i]:
                self.digits_right += 1

            
    def predict(self, dir):
        self.traverseFolders(self.process, dir, ftype='image')
        print float(self.digits_right)/self.digits_num


if __name__ == '__main__':
    test = Pytesser()
    test.predict('../../ocr_datasets/training_dir_groups')
