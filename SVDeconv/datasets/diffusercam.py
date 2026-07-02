from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)
from skimage.transform import resize as resize_ski


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

rng = np.random.default_rng(seed=42)

ex = Experiment("data")
ex = initialise(ex)


SIZE = 300,400 #270, 480

def region_of_interest(x):
    return x[..., 60:270, 60:440]

def resize_to_sensor( image, sensor_shape):
    #resize without changing the aspect ratio, crop if needed
    target_h, target_w = sensor_shape
    img_h, img_w = image.shape[:2]
    scale_h = img_h / target_h
    scale_w = img_w / target_w
    
    scale = min(scale_h,scale_w)
    new_h = int(target_h * scale)
    new_w = int(target_w * scale)

    crop_h = (img_h - new_h) // 2
    crop_w = (img_w - new_w) // 2

    image = image[crop_h:crop_h+new_h, crop_w:crop_w+new_w, ...]
    image = resize_ski(image, 
                (target_h, target_w),
                preserve_range=True,
                anti_aliasing=True).astype(np.float32)
    return image

def transform(image,working_size, gray=False,):
    # print(image.shape)
    image = np.flip(np.flipud(image), axis=2)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, working_size)
    image = (image - 0.5) * 2
    return image


def sort_key(x):
    return int(x[2:-4])


def load_psf(path,working_size):
    psf = np.array(Image.open(path))
    return transform(psf,working_size)


class LenslessLearning(Dataset):
    def __init__(self, diffuser_images, ground_truth_images,working_size,normalize=4095):
        """
        Everything is upside-down, and the colors are BGR...
        """
        self.xs = diffuser_images
        self.ys = ground_truth_images
        self.normalize = normalize
        self.working_size = working_size

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
            x = transform(x,working_size=self.working_size)
        elif diffused.name.endswith('.tiff'):
            x = cv2.imread(diffused, -1).astype(np.float32)/self.normalize
            x = transform(x,working_size=self.working_size)
        else:
            x = transform(np.load(diffused),working_size=self.working_size)
        
        if ground_truth.name.endswith('.png'):
            y = np.array(Image.open(ground_truth))
            y = transform(y,working_size=self.working_size)
        elif diffused.name.endswith('.tiff'):
            y= cv2.imread(ground_truth, -1).astype(np.float32)/self.normalize
            y = transform(y,working_size=self.working_size)
        else:
            y = transform(np.load(ground_truth),working_size=self.working_size)
        
        return x, y, str(diffused.name)


class LenslessLearningInTheWild(Dataset):
    def __init__(self, path,working_size,suffix='.npy',normalize=4095):
        xs = []
        self.suffix = suffix
        self.normalize = normalize
        self.working_size = working_size
        manifest = sorted((x.relative_to(path) for x in path.rglob(f'*{suffix}')))
        manifest = [f for f in manifest if f.parent.name != 'gt_tiff']
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
            x = transform(diffused,working_size=self.working_size)
        elif self.suffix in ('.tiff','.bmp'):            
            testim = cv2.imread(self.xs[idx], -1).astype(np.float32)/self.normalize #4095.#  - 0.008273973
            #testim = resize_to_sensor(testim,SIZE)
            testim = transform(testim,working_size=self.working_size)
            # testim = cv2.resize(testim, (480, 270))
            # testim = (testim - 0.5) * 2
            # testim= testim.transpose((2, 0, 1))
            # #testim = np.expand_dims(testim,0)
            # testim = torch.tensor(testim)

        return testim, torch.tensor(0), str(self.xs[idx].name)


class LenslessLearningCollection:
    def __init__(self, args):
        path = Path(args.image_dir)

        #self.psf = load_psf(path / 'psf.tiff')

        if args.train_csv_filename is not None:
            train_diffused, train_ground_truth = load_manifest(path,
                 csv_filename=args.train_csv_filename, 
                 decode_sim = args.decode_sim,
                 use_simulated_dataset=args.simulated_dir is not None,
                 simulated_dataset_dir=args.simulated_dir)

            if args.sanity_eval:
                train_diffused, train_ground_truth =[],[]
            val_diffused, val_ground_truth = load_manifest(path,
                         csv_filename=args.val_csv_filename, 
                        decode_sim = args.decode_sim, 
                        use_simulated_dataset=args.simulated_dir is not None,
                        simulated_dataset_dir=args.simulated_dir)

        else: 
            train_diffused, train_ground_truth, val_diffused, val_ground_truth = get_files_list_all_datasets(path,suffix='.tiff')

        self.train_dataset = LenslessLearning(train_diffused, train_ground_truth,
                                        working_size=(args.height,args.width),
                                        normalize=args.normalize_val)
        self.val_dataset = LenslessLearning(val_diffused, val_ground_truth,
                                        working_size=(args.height,args.width),
                                        normalize=args.normalize_val)
        if args.test_set_path is not None:
            self.test_dataset = LenslessLearningInTheWild(path / args.test_set_path,
                                    working_size=(args.height,args.width),
                                    suffix='.bmp',normalize=args.normalize_val)
        else:
            self.test_dataset = None
        self.region_of_interest = region_of_interest



def get_files_list_from_dir(path,suffix = '.tiff'):

    files_list =[p.name for p in (path/'gt_tiff').iterdir() if p.is_file() and p.suffix == suffix]   
    files_list = rng.permutation(files_list)

    
    train_files_names = files_list[:int(len(files_list)*0.9)]
    val_files_names = files_list[int(len(files_list)*0.9):]

    train_diffused_files = [path/'diffused'/x for x in train_files_names]
    train_gt_files = [path/'gt_tiff'/x for x in train_files_names]

    val_diffused_files = [path/'diffused'/x for x in val_files_names]
    val_gt_files = [path/'gt_tiff'/x for x in val_files_names]

    return train_diffused_files, train_gt_files, val_diffused_files, val_gt_files

def get_files_list_all_datasets(path,suffix = '.tiff'):
    train_diffused_files, train_gt_files, val_diffused_files, val_gt_files = [], [], [], []
    for dataset in path.iterdir():
        if dataset.is_dir():
            td, tg, vd, vg = get_files_list_from_dir(dataset,suffix)
            train_diffused_files.extend(td)
            train_gt_files.extend(tg)
            val_diffused_files.extend(vd)
            val_gt_files.extend(vg)
    return train_diffused_files, train_gt_files, val_diffused_files, val_gt_files


def load_manifest(path, csv_filename, decode_sim = False,use_simulated_dataset=False,simulated_dataset_dir=None):
    if csv_filename is not None:
        with open(path / csv_filename) as f:
            manifest = f.read().split()
    else:
        raise ValueError(f"csv_filename is required")


    xs, ys = [], []
    for filename in manifest:
        if use_simulated_dataset:
            x = Path(simulated_dataset_dir)/filename.replace(".jpg.tiff", ".tiff")
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