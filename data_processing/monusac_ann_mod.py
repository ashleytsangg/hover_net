'''
This script converts monusac labels to inst_map and type_map - this is for matching class labels with lymphocyte data

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

data_dir = "../../Downloads/MoNuSAC_images_and_annotations/"
ann_dir = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Monusac_Modified\Labels"

count = 0
for pt_folder in os.listdir(data_dir):
    for file in os.listdir(os.path.join(data_dir, pt_folder)):
        if file.endswith('.xml'):
            xml_file = os.path.join(data_dir, pt_folder) + '/' + file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            image_name = xml_file[:-4] + '.svs'
            print(image_name)
            img = openslide.OpenSlide(image_name)
            gt = 0

            binary_mask = np.transpose(
                np.zeros((img.read_region((0, 0), 0, img.level_dimensions[0]).size)))
            inst_map = np.copy(binary_mask)
            class_type = 0
            inst_type = 0

            # Generate binary mask for each cell-type
            for k in range(len(root)):
                label = [x.attrib['Name'] for x in root[k][0]]
                label = label[0]

                for child in root[k]:
                    for x in child:
                        r = x.tag
                        if r == 'Attribute':
                            count = count + 1
                            # print(count)
                            label = x.attrib['Name']

                            # binary_mask = np.transpose(
                            #     np.zeros((img.read_region((0, 0), 0, img.level_dimensions[0]).size)))
                            # print(label)
                            class_type = class_type + 1
                            if class_type == 1: # epithelial now class 6
                                class_label = 6
                            if class_type == 2: # lymphocyte now class 1
                                class_label = 1
                            # neutrophil stays as class 3


                        if r == 'Region':
                            if class_type != 4: # ignore the macrophage class - we don't want it
                                inst_type = inst_type + 1
                                regions = []
                                vertices = x[1]
                                coords = np.zeros((len(vertices), 2))
                                for i, vertex in enumerate(vertices):
                                    coords[i][0] = vertex.attrib['X']
                                    coords[i][1] = vertex.attrib['Y']
                                regions.append(coords)
                                poly = Polygon(regions[0])

                                vertex_row_coords = regions[0][:, 0]
                                vertex_col_coords = regions[0][:, 1]
                                fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords,
                                                                                binary_mask.shape)
                                # this is the type map
                                binary_mask[fill_row_coords, fill_col_coords] = class_label # label with new class label
                                # this is the inst map
                                inst_map[fill_row_coords, fill_col_coords] = inst_type


            ann_fn = ann_dir + "/" + file[:-4] + ".mat"
            mdict = {"inst_map": inst_map, "type_map": binary_mask}
            sio.savemat(ann_fn, mdict)
