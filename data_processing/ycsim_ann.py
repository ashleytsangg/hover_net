'''
This script converts YCSim labels to inst_map

'''

import os
import cv2

from xml.dom import minidom
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
import scipy
import scipy.ndimage
from shapely.geometry import Polygon
from skimage import draw
import xml.etree.ElementTree as ET

import openslide
from openslide import open_slide

data_dir = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\YCSim\Train\Im_Labels"
ann_dir = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\YCSim\Train\Labels"

count = 0
for file in os.listdir(data_dir):
    fn = os.path.join(data_dir, file)
    im = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)
    inst_map = np.array(im).astype('double')

    ann_fn = ann_dir + "/" + file[:9] + "Im" + file[13:-4] + ".mat"
    mdict = {"inst_map": inst_map}
    sio.savemat(ann_fn, mdict)

