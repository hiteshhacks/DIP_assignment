import argparse
import os
import sys

import cv2
import numpy as np

# Freeman 8-direction code: (dx, dy) -> code   (image y-axis points DOWN)
#   3 2 1
#   4 . 0      0=E 1=NE 2=N 3=NW 4=W 5=SW 6=S 7=SE
#   5 6 7
DIRS = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}


def make_test_image():
    img = np.full((360, 640, 3), 255, np.uint8)
    cv2.circle(img, (90, 100), 55, (0, 0, 200), -1)                      # circle
    cv2.rectangle(img, (190, 50), (310, 150), (200, 0, 0), -1)           # square
    cv2.fillPoly(img, [np.array([[450, 40], [390, 150], [510, 150]])],
                 (0, 130, 0))                                            # triangle
    cv2.circle(img, (100, 270), 60, (0, 0, 0), -1)                       # ring
    cv2.circle(img, (100, 270), 30, (255, 255, 255), -1)                 # 1 hole
    cv2.rectangle(img, (220, 210), (420, 330), (80, 80, 80), -1)         # 2 holes
    cv2.circle(img, (275, 270), 20, (255, 255, 255), -1)
    cv2.circle(img, (365, 270), 20, (255, 255, 255), -1)
    return img


class SyntheticVideo:
    """Fake camera: red ball bouncing left-right (used when no video is given)."""
    def __init__(self, n=600):
        self.i, self.n = 0, n

    def isOpened(self):
        return True

    def read(self):
        if self.i >= self.n:
            return False, None
        t, self.i = self.i, self.i + 1
        f = np.full((360, 640, 3), 230, np.uint8)
        x = int(60 + abs((t * 6) % 1040 - 520))
        y = int(180 + 100 * np.sin(t / 20))
        cv2.circle(f, (x, y), 22, (0, 0, 200), -1)
        return True, f

    def release(self):
        pass

def segment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:                      # objects must be white
        bw = 255 - bw
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

