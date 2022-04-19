from typing import Callable
import tifffile
import glob
import os
import numpy as np
import cv2

from DLIP.data.base_classes.base_dataset import BaseDataset


class BaseSegmentationDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        map_look_up: dict,
        samples_dir: str = "samples",
        class_dir: str = "labels",
        samples_data_format="tif",
        labels_data_format="tif",
        transforms = None,
        empty_dataset=False,
        labels_available=True,
        return_trafos=False,
        label_transform: Callable = None,
        labels_suffix = '_label',
        return_default_labels = False
    ):
        self.labels_available = labels_available
        self.root_dir = root_dir
        self.samples_dir = samples_dir
        self.class_dir = class_dir
        self.samples_data_format = samples_data_format
        self.labels_data_format = labels_data_format
        self.return_trafos = return_trafos
        self.transforms = transforms
        self.map_look_up = map_look_up
        self.label_transform = label_transform
        self.labels_suffix = labels_suffix
        self.default_class_dir = 'labels'
        self.return_default_labels = return_default_labels

        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]


        self.samples = os.path.join(self.root_dir,self.samples_dir)
        self.labels  = os.path.join(self.root_dir,self.class_dir)

        # Get all sample names sorted as integer values
        all_samples_sorted = sorted(
            glob.glob(f"{self.samples}{os.path.sep}*"),
            key=lambda x: 
                x.split(f"{self.samples}{os.path.sep}")[1].split(
                    f".{samples_data_format}"
            ),
        )
        self.indices = []
        if not empty_dataset:
            # Extract indices from the sorted samples
            self.indices = [
                i.split(f"{self.samples}{os.path.sep}")[1].split(f".{samples_data_format}")[0]
                for i in all_samples_sorted
            ]
        self.raw_mode = False

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # load sample
        sample_path = os.path.join(self.samples, f"{self.indices[idx]}.{self.samples_data_format}")
        sample_img = tifffile.imread(sample_path) if self.samples_data_format=="tif" else cv2.imread(sample_path,-1)

        sample_img_lst = []
        label_lst = []
        default_label_lst = []
        trafo_lst = []

        if self.labels_available:
            # load label map
            label_path = os.path.join(self.labels, f"{self.indices[idx]}{self.labels_suffix}.{self.labels_data_format}")
            label_img = tifffile.imread(label_path) if self.labels_data_format=="tif" else cv2.imread(label_path,-1)
            label_one_hot = label_img
            if self.label_transform is not None:
                label_one_hot = self.label_transform(label_one_hot)
        if self.return_default_labels:
            default_label_path = os.path.join(self.labels, f"{self.indices[idx]}{self.labels_suffix}.{self.labels_data_format}")
            default_label_path = default_label_path.replace(self.class_dir,self.default_class_dir)
            default_label_img = tifffile.imread(default_label_path) if self.labels_data_format=="tif" else cv2.imread(default_label_path,-1)
            default_label_one_hot = default_label_img
            if self.label_transform is not None:
                default_label_one_hot = self.label_transform(default_label_one_hot)

        # raw mode -> no transforms
        if self.raw_mode:
            if self.labels_available:
                return sample_img,label_one_hot
            else:
                return sample_img
            
        for transform in self.transforms:
            im, lbl, trafo = transform(sample_img, label_one_hot)
            if self.return_default_labels:
                _, d_lbl, _ = transform(sample_img, default_label_one_hot)
                default_label_lst.append(d_lbl)
            sample_img_lst.append(im)
            label_lst.append(lbl)
            trafo_lst.append(trafo)

        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
            label_lst = label_lst[0] if len(label_lst) > 0 else label_lst
            trafo_lst = trafo_lst[0] if len(trafo_lst) > 0 else trafo_lst
            default_label_lst = default_label_lst[0] if len(default_label_lst) > 0 else default_label_lst
       
        # sample_img_lst (optional: labels) (optional: trafos)
        if not self.return_trafos and not self.labels_available:
            return sample_img_lst
        if self.return_trafos and not self.labels_available:
            return sample_img_lst, trafo_lst
        if not self.return_trafos and self.labels_available and self.return_default_labels:
            return sample_img_lst, label_lst.unsqueeze(2) if len(label_lst.shape) == 2 else label_lst,  default_label_lst.unsqueeze(2) if len(default_label_lst.shape) == 2 else default_label_lst
        if not self.return_trafos and self.labels_available and not self.return_default_labels:
            return sample_img_lst, label_lst.unsqueeze(2) if len(label_lst.shape) == 2 else label_lst
        if self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst, trafo_lst

    def pop_sample(self, index):
        return self.indices.pop(index)

    def add_sample(self, new_sample):
        self.indices.append(new_sample)

    def get_samples(self):
        return self.indices
