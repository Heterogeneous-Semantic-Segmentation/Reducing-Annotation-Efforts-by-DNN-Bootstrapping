import os
import random
import tifffile
import cv2
import numpy as np

from DLIP.data.base_classes.base_pl_datamodule import BasePLDataModule
from DLIP.data.base_classes.segmentation.base_seg_dataset import BaseSegmentationDataset

class GenericSegmentationDataModule(BasePLDataModule):
    def __init__(
        self,
        root_dir: str,
        n_classes: int,
        batch_size = 1,
        dataset_size = 1.0,
        val_to_train_ratio = 0,
        initial_labeled_ratio= 1.0,
        train_transforms=None,
        train_transforms_unlabeled=None,
        val_transforms=None,
        test_transforms=None,
        return_unlabeled_trafos=False,
        num_workers=32,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
        labels_dir = 'labels'
    ):
        super().__init__(
            dataset_size=dataset_size,
            batch_size = batch_size,
            val_to_train_ratio = val_to_train_ratio,
            num_workers = num_workers,
            pin_memory = pin_memory,
            shuffle = shuffle,
            drop_last = drop_last,
            initial_labeled_ratio = initial_labeled_ratio,
        )
        self.root_dir = root_dir
        self.train_labeled_root_dir     = os.path.join(self.root_dir, "train")
        self.train_unlabeled_root_dir   = os.path.join(self.root_dir, "unlabeled")
        self.test_labeled_root_dir      = os.path.join(self.root_dir, "test")
        self.train_transforms = train_transforms
        self.train_transforms_unlabeled = (
            train_transforms_unlabeled
            if train_transforms_unlabeled is not None
            else train_transforms
        )
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.return_unlabeled_trafos = return_unlabeled_trafos
        self.labeled_train_dataset: BaseSegmentationDataset = None
        self.unlabeled_train_dataset: BaseSegmentationDataset = None
        self.val_dataset: BaseSegmentationDataset = None
        self.test_dataset: BaseSegmentationDataset = None
        self.n_classes = n_classes
        self.labels_dir = labels_dir
        self.samples_data_format, self.labels_data_format = 'tif','tif'
        self.map_lookup = {0: 1}
        self.__init_datasets()

    def __init_datasets(self):
        self.labeled_train_dataset = BaseSegmentationDataset(
            root_dir=self.train_labeled_root_dir, 
            transforms=self.train_transforms,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            class_dir=self.labels_dir,
            map_look_up=self.map_lookup
        )

        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
            self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))

        self.val_dataset = BaseSegmentationDataset(
            root_dir=self.train_labeled_root_dir, 
            transforms=self.val_transforms,
            empty_dataset=True,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            map_look_up=self.map_lookup,
            class_dir=self.labels_dir,
        )

        self.unlabeled_train_dataset = BaseSegmentationDataset(
            root_dir=self.train_unlabeled_root_dir,
            transforms=self.train_transforms,
            labels_available=False,
            return_trafos=self.return_unlabeled_trafos,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            map_look_up=self.map_lookup,
            class_dir=self.labels_dir,
        )
        
        self.test_dataset = BaseSegmentationDataset(
            root_dir=self.test_labeled_root_dir, 
            transforms=self.test_transforms,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            map_look_up=self.map_lookup,
            class_dir=self.labels_dir,
            return_default_labels = True
        )

    def _determine_data_format(self):
        extensions = {"samples": list(), "labels": list()}

        for folder in extensions.keys():
            for file in os.listdir(os.path.join(self.train_labeled_root_dir,folder)):
                extensions[folder].append(os.path.splitext(file)[1].replace(".", ""))

            for file in os.listdir(os.path.join(self.train_unlabeled_root_dir,folder)):
                extensions[folder].append(os.path.splitext(file)[1].replace(".", ""))

        return max(set(extensions["samples"]), key = extensions["samples"].count),max(set(extensions["labels"]), key = extensions["labels"].count)


    # def _determine_label_maps(self):
    #     map_lst = list()
    #     for file in os.listdir(os.path.join(self.train_labeled_root_dir,"labels")):
    #         file_path = os.path.join(self.train_labeled_root_dir,"labels", file)
    #         label_img = tifffile.imread(file_path) if self.labels_data_format=="tif" else cv2.imread(file_path,-1)
    #         map_lst.extend(np.unique(label_img))

    #     map_lst = sorted(map_lst)

    #     map_look_up = dict()
    #     if self.n_classes>1:
    #         for i in range(self.n_classes):
    #             map_look_up[i]= map_lst[i]
    #     else:
    #         map_look_up[0]= map_lst[-1]

    #     return map_look_up
