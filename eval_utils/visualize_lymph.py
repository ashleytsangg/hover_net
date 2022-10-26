'''
This script takes in the output .mat files, and computes evaluation metrics and outputs confusion matrix

'''

import os

import numpy as np
import scipy.io as sio
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

## set path to data directories

# path to out/eval dir of hovernet
pred_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\out\Lymphocyte\0921 data\full branch\pannuke_30_WLM0'
im_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\0921 dataset\All\Images'

eval_dir = pred_dir + '\eval'
mat_dir = pred_dir + '\mat'

save_dir = pred_dir + '\lymph_eval'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# loop through files
for fn in os.listdir(im_dir):
    im_name = fn[:-4]
    print('processing image ', im_name)
    # get pred type map
    pred_file = os.path.join(eval_dir, im_name + '_eval.mat')
    pred_matfile = sio.loadmat(pred_file)
    pred_tm = pred_matfile['type_map']
    pred_tm[pred_tm != 1] = 0
    pred_file_orig = os.path.join(mat_dir, im_name + '.mat')
    pred_matfile_orig = sio.loadmat(pred_file_orig)
    pred_instype = pred_matfile_orig['inst_type']
    pred_centroid = pred_matfile_orig['inst_centroid']
    x = np.where(pred_instype == 1)
    lymph_ind = np.squeeze(x)[0]
    lymph_centroid_x = pred_centroid[lymph_ind,0]
    lymph_centroid_y = pred_centroid[lymph_ind,1]
    plt.figure()
    plt.scatter(lymph_centroid_x, lymph_centroid_y, s=0.2, color='lime', marker='o')
    n_lymph = len(lymph_ind)
    plt.title('{}, lymph count: {}'.format(im_name, n_lymph))
    img = cv2.cvtColor(cv2.imread(os.path.join(im_dir, im_name + '.jpg')), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    # plt.imshow(pred_tm, alpha=0.6)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.savefig(os.path.join(save_dir, im_name + '.png'), dpi=800, bbox_inches='tight')
