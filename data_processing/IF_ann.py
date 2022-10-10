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
from skimage import io

import openslide
from openslide import open_slide

data_dir = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-05 YC Nuclear Segmentation of Breast Tumor TMA3 using Cellpose and Mesmer\Mesmer\B1\Mesmer model c1 stack to rgb"
ann_dir = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\IF\Train\Labels_B5_filt"

# count = 0
# for file in os.listdir(data_dir):
#     fn = os.path.join(data_dir, file)
#     if file == 'Thumbs.db':
#         continue
#     print(fn)
#     im = io.imread(fn)
#     inst_map = np.array(im).astype('double').squeeze()
#
#     ann_fn = ann_dir + "/" + file[:-4] + ".mat"
#     mdict = {"inst_map": inst_map}
#     sio.savemat(ann_fn, mdict)

file = 'TMA3_blob017_B01_IF1x_c1-crop-rgb_feature_1_filt.tif'
fn = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-05 YC Nuclear Segmentation of Breast Tumor TMA3 using Cellpose and Mesmer\Mesmer\B1\Mesmer model c1 stack to rgb\TMA3_blob017_B01_IF1x_c1-crop-rgb_feature_1_filt.tif'

im = io.imread(fn)
inst_map = np.array(im).astype('double').squeeze()

ann_fn = ann_dir + "/" + file[:-4] + ".mat"
mdict = {"inst_map": inst_map}
sio.savemat(ann_fn, mdict)


