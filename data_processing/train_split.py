"This script is for randomly shuffling train/test and saving images to corresponding folders"

import numpy as np
import os
import cv2

imdir = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\0921 dataset\All\Images"
split = 0
savedir = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\0921 dataset\Split_{}".format(split)
if not os.path.exists(savedir):
    os.mkdir(savedir)
traindir = os.path.join(savedir, 'Train\Images')
testdir = os.path.join(savedir, 'Test')

if not os.path.exists(traindir):
    os.mkdir(traindir)
if not os.path.exists(testdir):
    os.mkdir(testdir)

files = os.listdir(imdir)
if 'Thumbs.db' in files:
    files.remove('Thumbs.db')

num_im = len(files)
files = np.array(files)

split = round(0.7*num_im)
np.random.shuffle(files)
train_im = files[:split]
test_im = files[split:]

for im in train_im:
    fullim = os.path.join(imdir, im)
    img = cv2.imread(fullim)
    savenm = os.path.join(traindir, im)
    cv2.imwrite(savenm, img)

for im in test_im:
    fullim = os.path.join(imdir, im)
    img = cv2.imread(fullim)
    savenm = os.path.join(testdir, im)
    cv2.imwrite(savenm, img)
