import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

out_dir = 'out/Lymphocyte/pannuke_lymph_50_fix/'
mat_dir = out_dir + 'mat/'
json_dir = out_dir + 'json/'
raw_img_dir = 'dataset/Lymphocyte/Test/Images/'

for fn in os.listdir(json_dir):
    # read in json file
    file = os.path.join(json_dir, fn)
    f = open(file)
    data = json.load(f)

    # get list of contours and type
    contour_list = []
    type_list = []
    nuc_info = data['nuc']
    for inst in nuc_info:
        inst_info = nuc_info[inst]
        inst_contour = inst_info['contour']
        contour_list.append(inst_contour)
        inst_type = inst_info['type']
        type_list.append(inst_type)

    file = os.path.join(raw_img_dir, fn[:-5]) + '.jpg'
    im = cv2.imread(file)
    im = np.zeros_like(im)
    for c in contour_list:
        print(c)
        cv2.drawContours(im.astype('uint8'), [np.array(c)], -1, (255, 0, 0), -1)
    cv2.imwrite('countour.png', im)
    break