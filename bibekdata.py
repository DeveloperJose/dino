#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:24:56 2021

@author: mibook
"""
import glob, os, random, torch, elasticdeform, pathlib
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import pdb

from PIL import Image

band_names = np.array(["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7", "elevation", "slope"])

def fetch_loaders(processed_dir, batch_size=32, use_channels=[0,1], normalize=False, use_physics=False, train_folder='train', val_folder='val', test_folder='test', shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation
    Args:
        processed_dir(str): Directory with the processed data
        batch_size(int): The size of each batch during training. Defaults to 32.
    Return:
        Returns train and val dataloaders
    """
    train_dataset = GlacierDataset(processed_dir / train_folder, use_channels, normalize, use_physics,
                                   transforms=transforms.Compose([
                                       #DropoutChannels(0.5),
                                       FlipHorizontal(0.15),
                                       FlipVertical(0.15),
                                       Rot270(0.15),
                                       #ElasticDeform(1)
                                    ])
                                    )
    val_dataset = GlacierDataset(processed_dir / val_folder, use_channels, normalize, use_physics)
    test_dataset = GlacierDataset(processed_dir / test_folder, use_channels, normalize, use_physics)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              worker_init_fn=seed_worker, generator=g,
                              num_workers=8, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            worker_init_fn=seed_worker, generator=g,
                            num_workers=8, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            worker_init_fn=seed_worker, generator=g,
                            num_workers=8, shuffle=False)
    return train_loader, val_loader, test_loader


class GlacierDataset(Dataset):
    """Custom Dataset for Glacier Data
    Indexing the i^th element returns the underlying image and the associated
    binary mask
    """

    def __init__(self, folder_path, use_channels, normalize, use_physics, transforms=None):
        """Initialize dataset.
        Args:
            folder_path(str): A path to data directory
        """

        self.img_files = glob.glob(os.path.join(folder_path, '*tiff*'))
        self.mask_files = [s.replace("tiff", "mask") for s in self.img_files]
        self.phys_files = [s.replace("tiff", "physics_256") for s in self.img_files]

        self.use_channels = use_channels
        self.normalize = normalize
        self.use_physics = use_physics
        self.transforms = transforms
        arr = np.load(folder_path.parent / "normalize_train.npy")   

        if self.normalize == "min-max":
            self.min, self.max = arr[2][use_channels], arr[3][use_channels]
        if self.normalize == "mean-std":
            self.mean, self.std = arr[0], arr[1]
            self.mean, self.std = self.mean[use_channels], self.std[use_channels]

        print(f'Using channels {band_names[use_channels]} for {folder_path}')
        temp = np.load(self.img_files[0])
        assert temp.shape[2] == len(band_names), f'Length {len(band_names)} does not match shape[2] {temp.shape[2]} | full shape = {temp.shape}'

        # t1, t2 = self[0]
        # print(t1.shape, t2.shape)

    def __getitem__(self, index):
        """ getitem method to retrieve a single instance of the dataset
        Args:
            index(int): Index identifier of the data instance
        Return:
            data(x) and corresponding label(y)
        """
        file_data = np.load(self.img_files[index])
        data = file_data[:, :, self.use_channels]

        # _mask = np.sum(data, axis=2) == 0
        if self.normalize == "min-max":
            data = np.clip(data, self.min, self.max)
            data = (data - self.min) / (self.max - self.min)
        elif self.normalize == "mean-std":
            data = (data - self.mean) / self.std
        else:
            raise ValueError("normalize must be min-max or mean-std")
        # label = np.expand_dims(np.load(self.mask_files[index]), axis=2)
        # #ones = label == 1
        # #twos = label == 2
        # #zeros = np.invert(ones + twos)
        # #label = np.concatenate((zeros, ones, twos), axis=2)
        # label = np.concatenate((label == 0, label == 1), axis=2)
        # label[_mask] = 0

        if self.use_physics:
            im_phys = np.load(self.phys_files[index])
            temp = np.zeros((data.shape[0], data.shape[1], len(self.use_channels)+1))
            temp[:, :, :-1] = data
            temp[:, :, -1] = im_phys
            data = temp

        if self.transforms:
            # print(data.shape, data.dtype)
            data = Image.fromarray(data, 'RGB')
            data = self.transforms(data)
            # sample = {'image': data, 'mask': label}
            # sample = self.transforms(sample)
            # data = torch.from_numpy(sample['image'].copy()).float()
            # label = torch.from_numpy(sample['mask'].copy()).float()

        # else:
        #     data = torch.from_numpy(data).float()
        #     label = torch.from_numpy(label).float()
        return data, torch.Tensor()

    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)
        """
        return len(self.img_files)


class FlipHorizontal(object):
    """Flip horizontal randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipHorizontal
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data[:, ::-1, :]
            label = label[:, ::-1, :]
        return {'image': data, 'mask': label}


class FlipVertical(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipVertical
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data[::-1, :, :]
            label = label[::-1, :, :]
        return {'image': data, 'mask': label}


class Rot270(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of Rot270
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data.transpose((1, 0, 2))
            label = label.transpose((1, 0, 2))
        return {'image': data, 'mask': label}

class DropoutChannels(object):
    """
    Apply Random channel dropouts
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            rand_channel_index = np.random.randint(low=0, high=data.shape[2], size=int(data.shape[2]/5))
            data[:, :, rand_channel_index] = 0
        return {'image': data, 'mask': label}

class ElasticDeform(object):
    """
    Apply Elasticdeform from U-Net
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        label = label.astype(np.float32)
        if torch.rand(1) < self.p:
            [data, label] = elasticdeform.deform_random_grid([data, label], axis=(0, 1))
        label = np.round(label).astype(bool)
        return {'image': data, 'mask': label}