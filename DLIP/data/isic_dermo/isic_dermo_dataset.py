import glob
import os
import cv2
import random
import torch
import numpy as np

from DLIP.data.base_classes.base_dataset import BaseDataset

class IsicDermoDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        samples_data_format="jpg",
        labels_data_format="png",
        transforms=None,
        empty_dataset=False,
        insert_bg_class=False,
        labels_available=True,
        return_trafos=False,
        labels_dir = 'labels',
        return_default_labels = False
    ):
        self.root_dir = root_dir
        self.labels_dir_name = labels_dir
        self.samples_data_format = samples_data_format
        self.labels_data_format = labels_data_format
        self.labels_dir = os.path.join(self.root_dir, labels_dir)
        self.samples_dir = os.path.join(self.root_dir, 'samples_256')
        self.insert_bg_class = insert_bg_class
        self.labels_available = labels_available
        self.return_trafos = return_trafos
        self.transforms = transforms
        self.return_default_labels = return_default_labels
        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]

        all_samples_sorted = sorted(
            glob.glob(f"{self.samples_dir}{os.path.sep}*.{self.samples_data_format}"),
            key=lambda x: int(x.split(f'.{self.samples_data_format}')[0].split('_')[-1]),
        )
        self.indices = []
        if not empty_dataset:
            self.indices = [i.split(f'.{self.samples_data_format}')[0].split('_')[-1] for i in all_samples_sorted]
        
        self.raw_mode = False
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_img = np.array(
            cv2.imread(os.path.join(self.samples_dir,
                f"ISIC_{self.indices[idx]}.{self.samples_data_format}"),
                -1,
            )
        )

        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

        label = None
        if self.labels_available:
            label_path = os.path.join(
                self.labels_dir, 
                f"ISIC_{self.indices[idx]}_segmentation.{self.labels_data_format}"
            )
            label = cv2.imread(label_path, -1)
            label = np.where(label == 0, 0,1)
            label = np.expand_dims(label,2)
            if self.insert_bg_class:
                # convert to one hot encoding
                label = np.stack((label,np.where(label,0,1)))
            if self.return_default_labels:
                default_label_path = os.path.join(
                    self.labels_dir, 
                    f"ISIC_{self.indices[idx]}_segmentation.{self.labels_data_format}"
                ).replace(self.labels_dir_name,'labels')
                default_label = cv2.imread(default_label_path, -1)
                default_label = np.where(default_label == 0, 0,1)
                default_label = np.expand_dims(default_label,2)
                if self.insert_bg_class:
                    # convert to one hot encoding
                    default_label = np.stack((default_label,np.where(default_label,0,1)))

        # raw mode -> no transforms
        if self.raw_mode:
            if self.labels_available:
                return sample_img,label
            else:
                return sample_img
        
        sample_img_lst = []
        label_lst = []
        default_lbl_lst = []
        trafo_lst = []
        for transform in self.transforms:
            im, lbl, trafo = transform(sample_img, label)
            if self.return_default_labels:
                _,d_lbl,_ = transform(sample_img,default_label)
                default_lbl_lst.append(d_lbl)
            sample_img_lst.append(im)
            label_lst.append(lbl)
            trafo_lst.append(trafo)

        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
            label_lst = label_lst[0] if len(label_lst) > 0 else label_lst
            trafo_lst = trafo_lst[0] if len(trafo_lst) > 0 else trafo_lst
            default_lbl_lst = default_lbl_lst[0] if len(default_lbl_lst) > 0 else default_lbl_lst
        
        if not self.return_trafos and not self.labels_available:
            return sample_img_lst
        if not self.return_trafos and self.labels_available and self.return_default_labels:
            return sample_img_lst, label_lst,default_lbl_lst
        if not self.return_trafos and self.labels_available and not self.return_default_labels:
            return sample_img_lst, label_lst
        if not self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst.float() if isinstance(label_lst,torch.Tensor) else label_lst
        if self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst.float() if isinstance(label_lst,torch.Tensor) else label_lst, trafo_lst
    
    def get_samples(self):
        return self.indices

    def pop_sample(self, index):
        return self.indices.pop(index)

    def add_sample(self, new_sample):
        return self.indices.append(new_sample)
