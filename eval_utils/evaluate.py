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

pred_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\out\Lymphocyte\pannuke_finetuning\pannuke_lymph_50'
gt_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\Additional\Ground Truth'

pred_mat_dir = os.path.join(pred_dir, 'mat/')

# assumption: filenames of prediction and gt are the same

for fn in os.listdir(pred_mat_dir):
    ## read in the matfile of the prediction
    pred_file = os.path.join(pred_mat_dir, fn)
    pred_matfile = sio.loadmat(pred_file)
    # get inst type of pred
    inst_uid = pred_matfile['inst_uid']
    inst_type = pred_matfile['inst_type']
    inst_map = pred_matfile['inst_map']
    inst_type = pred_matfile['inst_type']
    ## get type map
    type_map = inst_map
    for uid in inst_uid:
        uid = uid[0]  # this is the uid number, starting at 1
        mask = inst_map == uid
        type_map[mask] = inst_type[uid - 1][0]  # index inst_type by index, not uid number

    ## read in the matfile of the ground truth
    fn = 'CellCounter_' + fn[:-3] + 'csv'
    gt_file = os.path.join(gt_dir, fn)
    csvfile = pd.read_csv(gt_file)
    imsize = inst_map.shape
    y_coord = np.array(csvfile['Y'].values)
    x_coord = np.array(csvfile['X'].values)
    gt_type = np.array(csvfile['Type'].values)
    inst_map_sub = type_map[y_coord, x_coord]
    ## get predictions and gt
    not_predicted = inst_map_sub == 0
    pred = inst_map_sub[~not_predicted].squeeze()
    # TODO: i think can get the inst type by adjusting the coord (don't need to make the type map)
    # pred = inst_type[y_coord-1, x_coord-1]
    # pred = pred[~not_predicted].squeeze()
    truth = gt_type[~not_predicted]

    acc = np.sum(pred == truth) / len(pred)

    cm = confusion_matrix(truth, pred)

    g = sn.heatmap(cm, annot=True, fmt='.5g')
    plt.xlabel('Pred Class')
    plt.ylabel('True Class')
    # check if 0 in prediction (if background predicted)
    if 0 in pred:
        lower = 0
    else:
        lower = 1
    # get number of classes
    upper = np.max(truth)
    g.set_xticklabels(np.arange(lower, upper))
    plt.title('FTE302-20X-STIC, acc ={}%'.format(acc))

    assert False







