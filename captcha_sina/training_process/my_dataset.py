import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

import string

TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'train'

class mydataset(Dataset):

    def __init__(self, folder):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        
        return image, label

import string
# characters = '-' + string.digits + string.ascii_uppercase
characters = '-' + string.digits + string.ascii_lowercase
width, height, n_len, n_classes = 100, 40, 6, len(characters)
n_input_length = 12
print(characters, width, height, n_len, n_classes)


class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length,folder):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image_root = self.train_image_file_paths[index]
        image_name = image_root.split(os.path.sep)[-1]
        image = to_tensor(Image.open(image_root))
        random_str = image_name.split('_')[0]
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length
       

dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len,folder=TRAIN_DATASET_PATH)
image, target, input_length, label_length = dataset[0]
print(''.join([characters[x] for x in target]), input_length, label_length)
to_pil_image(image)

