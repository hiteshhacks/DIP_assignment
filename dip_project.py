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

