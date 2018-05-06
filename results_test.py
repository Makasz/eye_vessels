import cv2
import numpy as np
from os import listdir

class Img:
    def __init__(self, img):
        self.img = img
        self.count_white = cv2.countNonZero(img)
        self.count_black = cv2.countNonZero(cv2.bitwise_not(img))
        self.size = self.count_white + self.count_black

    def compare(self, other_img):
        out = cv2.bitwise_and(self.img, other_img.img)
        self.proper_hits = cv2.countNonZero(out)
        self.wrong_hits = self.count_white - self.proper_hits
        self.no_hits = other_img.count_white - self.proper_hits

    def show_results(self):
        print("Marked pixels: " + str(self.count_white))
        print("Proper hits: " + str(self.proper_hits))
        print("Wrong hits: " + str(self.wrong_hits))
        print("Not hitted pixels: " + str(self.no_hits))

OUR = 'result.png'
PROF = 'expected_result.tif'

img_our = cv2.imread(OUR, 0)
img_prof = cv2.imread(PROF, 0)

im1 = Img(img_our)
im2 = Img(img_prof)

im1.compare(im2)
im1.show_results()
