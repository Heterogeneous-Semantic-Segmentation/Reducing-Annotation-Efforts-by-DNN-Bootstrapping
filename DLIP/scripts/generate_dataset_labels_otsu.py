
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.cluster
from skimage import measure
from tqdm import tqdm
from os.path import exists
import os
from glob import glob
import tifffile
from skimage import filters
from skimage import exposure

SAMPLES_DIR = '/home/ws/kg2371/datasets/2017_ISIC_Derma/train/samples'
NUM_CLUSTERS = 3

samples = sorted(glob(os.path.join(SAMPLES_DIR,'*')),key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))
samples = glob(os.path.join(SAMPLES_DIR,'*'))
for img_p in tqdm(samples):
    i = img_p.split('/')[-1].split('.')[0]
    if not exists(img_p):
        continue
    if exists(f'/home/ws/kg2371/datasets/2017_ISIC_Derma/test/labels_otsu/{i}_segmentation.tif'):
        continue
    img = cv2.imread(img_p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        img = cv2.medianBlur(img,311)
    except Exception as e:
        print(e)
    #img = tifffile.imread(img_p)
    original_shape = img.shape[0:2]
    img = cv2.resize(img, (512,512), interpolation= cv2.INTER_LINEAR)
    img = cv2.medianBlur(img,11)
    
    val = filters.threshold_otsu(img)
    mask = (img < val)
    #tifffile.imwrite(f'/home/ws/kg2371/datasets/2017_ISIC_Derma/train/labels_otsu/{i}_label.tif',mask.astype(np.uint8))
    cv2.imwrite(f'/home/ws/kg2371/datasets/2017_ISIC_Derma/test/labels_otsu/{i}_segmentation.tif',(mask*255).astype(np.uint8))