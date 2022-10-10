import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import scipy.io as sio
from skimage import io


he_im = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\IF\Images\TMA3_blob017_B01.tif'
if_im = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-05 YC Nuclear Segmentation of Breast Tumor TMA3 using Cellpose and Mesmer\cropped raw img\B1 crop\TMA3_blob017_B01_IF1x_c1-crop-rgb.tif'
mesmer_mask = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\IF\Im_Labels\TMA3_blob017_B01.tif'
hov_out_mat = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\out\IF\ft_seg_10_np\mat\TMA3_blob017_B01.mat'

he_im = cv2.cvtColor(cv2.imread(he_im), cv2.COLOR_BGR2RGB)
if_im = cv2.cvtColor(cv2.imread(if_im), cv2.COLOR_BGR2GRAY)
if_im = np.array(if_im)
mesmer_mask = io.imread(mesmer_mask)
mes_mask = np.array(mesmer_mask).squeeze()
mes_mask[mes_mask > 0] = 1

hov_tm = sio.loadmat(hov_out_mat)['inst_map']
hov_tm = np.array(hov_tm)
hov_tm[hov_tm > 0] = 1

plt.subplot(2,2,1)
plt.title('raw H&E')
plt.imshow(he_im, interpolation='none')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.subplot(2,2,2)
plt.title('IF')
plt.imshow(he_im, interpolation='none')
plt.imshow(if_im,  alpha=0.5, interpolation='none')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.subplot(2,2,3)
plt.title('Mesmer Label')
plt.imshow(he_im, interpolation='none')
plt.imshow(mes_mask, alpha=0.3, interpolation='none')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.subplot(2,2,4)
plt.title('Hover Output')
plt.imshow(he_im, interpolation='none')
plt.imshow(hov_tm, alpha=0.3, interpolation='none')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.savefig('test.png', dpi=300)