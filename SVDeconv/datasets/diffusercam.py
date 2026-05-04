from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING
from sacred import Experiment

# Torch modules
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.distributed as dist
import os
import cv2
import numpy as np
from config import initialise
from pathlib import Path

if TYPE_CHECKING:
    from utils.typing_alias import *


ex = Experiment("data")
ex = initialise(ex)


SIZE = 270, 480

def region_of_interest(x):
    return x[..., 60:270, 60:440]


def transform(image, gray=False):
    # print(image.shape)
    image = np.flip(np.flipud(image), axis=2)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    image = (image - 0.5) * 2
    return image


def sort_key(x):
    return int(x[2:-4])


def load_psf(path):
    psf = np.array(Image.open(path))
    return transform(psf)


class LenslessLearning(Dataset):
    def __init__(self, diffuser_images, ground_truth_images):
        """
        Everything is upside-down, and the colors are BGR...
        """
        self.xs = diffuser_images
        self.ys = ground_truth_images

    def read_image(self, filename):
        image = np.load(filename)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        diffused = self.xs[idx]
        ground_truth = self.ys[idx]
        # print(diffused, ground_truth)
        # print("hello!", np.load(diffused).shape, np.load(ground_truth).shape)
        
        if diffused.name.endswith('.png'):
            x = np.array(Image.open(diffused))
            x = transform(x)
        else:
            x = transform(np.load(diffused))
        
        if ground_truth.name.endswith('.png'):
            y = np.array(Image.open(ground_truth))
            y = transform(y)
        else:
            y = transform(np.load(ground_truth))
        
        return x, y, str(diffused.name)


class LenslessLearningInTheWild(Dataset):
    def __init__(self, path,suffix='.npy'):
        xs = []
        self.suffix = suffix
        manifest = sorted((x.name for x in path.glob(f'*{suffix}')))
        for filename in manifest:
            xs.append(path / filename)

        self.xs = xs

    def read_image(self, filename):
        image = np.load(filename,allow_pickle=True)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if self.suffix == '.npy':
            diffused = self.read_image(self.xs[idx])
            x = transform(diffused)
        elif self.suffix == '.tiff':            
            testim = cv2.imread(self.xs[idx], -1).astype(np.float32)/4095.#  - 0.008273973
            testim = transform(testim)
            # testim = cv2.resize(testim, (480, 270))
            # testim = (testim - 0.5) * 2
            # testim= testim.transpose((2, 0, 1))
            # #testim = np.expand_dims(testim,0)
            # testim = torch.tensor(testim)

        return testim, torch.tensor(0), str(self.xs[idx].name)


class LenslessLearningCollection:
    def __init__(self, args):
        path = Path(args.image_dir)

        self.psf = load_psf(path / 'psf.tiff')

        train_diffused, train_ground_truth = load_manifest(path,
                 'dataset_train.csv', 
                 decode_sim = args.decode_sim,
                 use_simulated_dataset=args.simulated_dir is not None,
                 simulated_dataset_dir=args.simulated_dir)

        if args.sanity_eval:
            train_diffused, train_ground_truth =[],[]
        val_diffused, val_ground_truth = load_manifest(path, 'dataset_test.csv', 
                        decode_sim = args.decode_sim, 
                        use_simulated_dataset=args.simulated_dir is not None,
                        simulated_dataset_dir=args.simulated_dir)

        self.train_dataset = LenslessLearning(train_diffused, train_ground_truth)
        self.val_dataset = LenslessLearning(val_diffused, val_ground_truth)
        if args.test_set_path is not None:
            self.test_dataset = LenslessLearningInTheWild(path / args.test_set_path,suffix='.tiff')
        else:
            self.test_dataset = None
        self.region_of_interest = region_of_interest


def load_manifest(path, csv_filename, decode_sim = False,use_simulated_dataset=False,simulated_dataset_dir=None):
    with open(path / csv_filename) as f:
        manifest = f.read().split()

    xs, ys = [], []
    for filename in manifest:
        if use_simulated_dataset:
            x = Path(simulated_dataset_dir)/filename.replace(".jpg.tiff", ".png")
        else:
            x = path / 'diffuser_images' / filename.replace(".jpg.tiff", ".npy")
        if decode_sim:
            y = path / 'decode_sim_padding_png' / filename.replace(".jpg.tiff", ".png")
        else:
            y = path / 'ground_truth_lensed' / filename.replace(".jpg.tiff", ".npy")
        # if x.exists() and y.exists():
        #     print(f"Found {x} and {y}")
        xs.append(x)
        ys.append(y)
        # else:
        #     print(f"No file named {x}")
    # check all files exist
    for idx, (x, y) in enumerate(zip(xs, ys)):
        if not x.exists() or not y.exists():
            xs[idx] = None
            ys[idx] = None

    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    return xs, ys