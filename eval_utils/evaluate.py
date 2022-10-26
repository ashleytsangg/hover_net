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

pred_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\out\Lymphocyte\0921 data\full branch\pannuke_10_WLM00'
gt_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\0921 dataset\All\Raw Labels'

nr_types = 6

eval_save_dir = os.path.join(pred_dir, 'eval/')
if not os.path.exists(eval_save_dir):
    os.mkdir(eval_save_dir)
pred_mat_dir = os.path.join(pred_dir, 'mat/')

# assumption: filenames of prediction and gt are the same

result_dict = {}
# aggregate confusion matrix
agg_cm = np.zeros((nr_types, nr_types))

for fn in os.listdir(pred_mat_dir):
    ## read in the matfile of the prediction
    pred_file = os.path.join(pred_mat_dir, fn)
    im_name = fn[:-4]
    pred_matfile = sio.loadmat(pred_file)
    # get inst type of pred
    inst_uid = pred_matfile['inst_uid']
    inst_type = pred_matfile['inst_type']
    inst_map = pred_matfile['inst_map']
    inst_type = pred_matfile['inst_type']
    ## get type map
    print('{}: starting generating type map...'.format(im_name))
    type_map = inst_map
    for uid in inst_uid:
        uid = uid[0]  # this is the uid number, starting at 1
        mask = inst_map == uid
        type_map[mask] = inst_type[uid - 1][0]  # index inst_type by index, not uid number
    print('finished.')
    result_dict['type_map'] = type_map
    ## read in the matfile of the ground truth
    fn = 'CellCounter_' + im_name + '.csv'
    gt_file = os.path.join(gt_dir, fn)
    csvfile = pd.read_csv(gt_file)
    imsize = inst_map.shape
    y_coord = np.array(csvfile['Y'].values)
    x_coord = np.array(csvfile['X'].values)
    gt_type = np.array(csvfile['Type'].values)
    inst_map_sub = type_map[y_coord, x_coord]
    ## get predictions and gt
    # not_predicted = inst_map_sub == 0
    # pred = inst_map_sub[~not_predicted].squeeze()
    # truth = gt_type[~not_predicted]
    pred = inst_map_sub.squeeze()
    truth = gt_type
    ## get accuracy and output confusion matrix
    acc = np.sum(pred == truth) / len(pred)
    result_dict['accuracy'] = acc

    cm = confusion_matrix(truth, pred)

    # ensure all confusion matrices have same dimensions
    for type in range(nr_types):
        if (type not in pred) and (type not in truth):
            # pad with zeros
            cm = np.insert(cm, type, 0, axis=1)
            cm = np.insert(cm, type, 0, axis=0)

    result_dict['conf_mat'] = cm

    plt.figure()
    g = sn.heatmap(cm, annot=True, fmt='.5g')
    plt.xlabel('Pred Class')
    plt.ylabel('True Class')
    plt.title('{}, acc ={:.2f}%'.format(im_name, acc*100))
    ## save figures for each image
    plt.savefig(os.path.join(eval_save_dir, '{}_CM.png'.format(im_name)))
    sio.savemat(os.path.join(eval_save_dir, '{}_eval.mat'.format(im_name)), mdict=result_dict)

    # add  to aggregate confusion matrix
    agg_cm = agg_cm + cm

agg_result_dict = {}
agg_result_dict['conf_mat'] = agg_cm
# compute aggregate accuracy
agg_acc = np.trace(agg_cm) / np.sum(agg_cm)
agg_result_dict['accuracy'] = agg_acc

plt.figure()
g = sn.heatmap(agg_cm, annot=True, fmt='.5g')
plt.xlabel('Pred Class')
plt.ylabel('True Class')
plt.title('aggregate acc ={:.2f}%'.format(agg_acc*100))
## save figures for each image
plt.savefig(os.path.join(eval_save_dir, 'aggregate_CM.png'))
sio.savemat(os.path.join(eval_save_dir, 'aggregate_eval.mat'), mdict=agg_result_dict)







