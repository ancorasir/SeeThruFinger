from typing import Optional, Tuple
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image
import numpy as np
from skimage import io

import torchvision
import matplotlib.pyplot as plt

class Bone_finger_test(Dataset):
    """skin finger dataset."""

    def __init__(self, image_file,force_file, transform=None,finger=9):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): contains all the relative path Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """   

        self.images = np.load(image_file)
        if self.images.shape[2] == 640:
            self.images = self.images[:,:,140:500]

        if len(self.images.shape)==3:
            self.images = np.expand_dims(self.images,axis=3)
        self.labels = np.load(force_file)
        self.template = np.load('/home/fang/Documents/Track-Anything/cam%s_template.npy'%finger)[:,140:500].reshape([360,360,1])/255
        
        self.scale_y = [10, 10, 2, 1, 1, 0.2]
        for i in range(6):
            self.labels[:,i] = self.labels[:,i]/self.scale_y[i]
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_numpy = np.concatenate([self.images[idx],self.template],axis=2)

        labels = self.labels[idx]
        labels = labels.astype('float32').reshape(-1, 6)
        labels_tensor = torch.from_numpy(labels)

        if self.transform:
            image_tensor = self.transform(image_numpy)
        else :
            a = transforms.ToTensor()
            image_tensor = a(image_numpy)

        return (image_tensor, labels_tensor)

class Bone_finger(Dataset):
    """skin finger dataset."""

    def __init__(self, data_dir=[''], transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): contains all the relative path Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """   
        self.images_ = []
        self.labels_ = []
        self.template_cam9 = np.load('/home/fang/Documents/Track-Anything/cam9_template.npy')[:,140:500].reshape([360,360,1])/255
        self.template_cam10 = np.load('/home/fang/Documents/Track-Anything/cam10_template.npy')[:,140:500].reshape([360,360,1])/255

        for folder in data_dir:  
            image_file = folder+'/mask.npy'
            force_file = folder+'/force_vecs.npy'
            self.images_.append( np.load(image_file)[:,:,140:500].reshape([10000,360,360,1]) )
            self.labels_.append( np.load(force_file) )

        self.images_ = np.concatenate(self.images_, axis=0)
        self.labels_ = np.concatenate(self.labels_, axis=0)

        train_index = np.random.choice(40000, 32000, replace=False)
        mask = np.zeros(40000,dtype=bool)
        mask[train_index] = True

        self.images = self.images_[mask]
        self.labels = self.labels_[mask]

        np.save('test_images.npy',self.images_[~mask])
        np.save('test_labels.npy',self.labels_[~mask])
        np.save('test_index.npy',~mask)
        
        scale_y = [10, 10, 2, 1, 1, 0.2]
        for i in range(6):
            self.labels[:,i] = self.labels[:,i]/scale_y[i]
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # image_numpy = self.images[idx]
        if idx<20000:
            image_numpy = np.concatenate([self.images[idx],self.template_cam9],axis=2)
        else:
            image_numpy = np.concatenate([self.images[idx],self.template_cam10],axis=2)

        labels = self.labels[idx]
        labels = labels.astype('float32').reshape(-1, 6)
        labels_tensor = torch.from_numpy(labels)

        if self.transform:
            image_tensor = self.transform(image_numpy)
        else :
            a = transforms.ToTensor()
            image_tensor = a(image_numpy)

        return (image_tensor, labels_tensor)

class SoftFingerDataModule(LightningDataModule):
    """
    Example of LightningDataModule for SoftFinger dataset.
    SoftFinger dataset sample contains :
                 a reseized (224 * 224) image
                 a labels Fx, Fy, Fz, Tx, Ty, Tz
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: list = ["/dataset/"],
        train_test_split: Tuple[float, float] = (0.7, 0.2),
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize([224, 224]),
            ]
        )

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 224, 224)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
        # DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
        # print("Data mean", DATA_MEANS)
        # print("Data std", DATA_STD)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`."""
        if not self.data_train or not self.data_val or not self.data_test:
            dataset = Bone_finger(self.data_dir, transform=self.transform)

            train_length = int(
                self.train_test_split[0] * dataset.__len__())
            test_length = int(
                self.train_test_split[1] * dataset.__len__())
            val_length = int(dataset.__len__() -
                             train_length - test_length)

            self.data_train, self.data_val, self.data_test = random_split(
                dataset, (train_length, val_length,
                          test_length), generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def get_train_images(self, num):  
        return torch.stack([self.data_train[i][0] for i in range(num)], dim=0)

if __name__ == "__main__":
    a = SoftFingerDataModule()
    a.setup()

    for batch in a.train_dataloader():
        x, label = batch
        print(label.dtype)

        b = a.get_train_images(4)
        print(b.shape)
    