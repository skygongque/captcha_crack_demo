import os
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from captcha.image import ImageCaptcha
import string
import random
# DATASET_PATH
# TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'train'
TRAIN_DATASET_PATH = r'captcha_sina\training_process\dataset\train'

class SinaDataset(Dataset):
    # 数据集的length由folder中图片数决定
    def __init__(self, characters, width, height, input_length, label_length,folder):
        super(SinaDataset, self).__init__()
        self.characters = characters
        # self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.length = len(self.train_image_file_paths)

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


class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        image = to_tensor(self.generator.generate_image(random_str))
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length


if __name__ == "__main__":
    # 测试数据集
    characters = '-' + string.digits + string.ascii_lowercase
    width, height, n_len, n_classes = 100, 40, 6, len(characters)
    n_input_length = 12
    print(characters, width, height, n_len, n_classes)
    dataset = SinaDataset(characters, width, height, n_input_length, n_len,TRAIN_DATASET_PATH)
    print('dataset.length',dataset.length,'dataset.label_length',dataset.label_length)
    image, target, input_length, label_length = dataset[0]
    print(''.join([characters[x] for x in target]), input_length, label_length)
    to_pil_image(image).show()

