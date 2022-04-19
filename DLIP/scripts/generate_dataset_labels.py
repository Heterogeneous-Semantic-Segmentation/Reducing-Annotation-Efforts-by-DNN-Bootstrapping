
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

SAMPLES_DIR = '/home/ws/kg2371/datasets/2017_ISIC_Derma/train/samples'
NUM_CLUSTERS = 2

samples = sorted(glob(os.path.join(SAMPLES_DIR,'*')),key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))
for img_p in tqdm(samples):
    i = img_p.split('/')[-1].split('.')[0]
    if not exists(img_p):
        continue
    img = cv2.imread(img_p)
    #img = tifffile.imread(img_p)
    original_shape = img.shape[0:2]
    img = cv2.resize(img, (512,512), interpolation= cv2.INTER_LINEAR)
    img = cv2.medianBlur(img,31)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = NUM_CLUSTERS
    compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # most frequent label is assumed to be background
    labels = labels.flatten()
    background = np.argmax(np.bincount(labels))

    # set bg to -1 since its no valid label
    labels[labels==background] = -1
    labels[labels != -1] = 1
    labels[labels==-1] = 0

    seg_mask = labels.reshape(512,512)

    labels_mask = measure.label(seg_mask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    mask = labels_mask

    mask_big =  cv2.resize(mask, original_shape[::-1], 0, 0, interpolation = cv2.INTER_NEAREST)

    cv2.imwrite(f'/home/ws/kg2371/datasets/2017_ISIC_Derma/train/labels_generated_2/{i}_segmentation.tif',(mask_big*255).astype(np.uint8))
    #tifffile.imwrite(f'/home/ws/kg2371/datasets/2017_ISIC_Derma/test/labels_generated_2/{i}_segmentation.tif',(mask_big).astype(np.uint8))
    
    #cv2.imshow('Dilated Image', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()