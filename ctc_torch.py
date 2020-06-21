""" 
from: https://github.com/ypwhs/captcha_break 
对captcha生成的准确率特别高，几乎100%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from captcha.image import ImageCaptcha
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

import string

characters = '-' + string.digits + string.ascii_uppercase
# 验证码的长，宽和字符数
width, height, n_len, n_classes = 192, 64, 4, len(characters)
n_input_length = 12
print(characters, width, height, n_len, n_classes)


# 搭建数据集 用captcha库生成验证码
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
        random_str = ''.join([random.choice(self.characters[1:])
                              for j in range(self.label_length)])
        # to_tensor把图片转为tensor类型
        image = to_tensor(self.generator.generate_image(random_str))
        target = torch.tensor([self.characters.find(x)
                               for x in random_str], dtype=torch.long)
        input_length = torch.full(
            size=(1, ), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(
            size=(1, ), fill_value=self.label_length, dtype=torch.long)
        # image为tensor类型
        return image, target, input_length, target_length


dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)
image, target, input_length, label_length = dataset[0]
print(''.join([characters[x] for x in target]), input_length, label_length)
pic = to_pil_image(image)
# pic.show()


batch_size = 128
train_set = CaptchaDataset(
    characters, 1000 * batch_size, width, height, n_input_length, n_len)
valid_set = CaptchaDataset(
    characters, 100 * batch_size, width, height, n_input_length, n_len)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=12)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=12)

# 解码函数和准确率计算函数


def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1])
                 if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')


def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true) == decode(pred)
                  for true, pred in zip(target, output_argmax)])
    return a.mean()


# 神经网络模型
class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        super(Model, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()

        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block+1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size=self.infer_features(
        ), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,)+self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# torch加载模型前要定义好相同的模型（参数未优化），加载模型相当于加载优化后的参数
# 加载模型 映射到cpu-only的设备
device = torch.device('cpu')
model = torch.load('ctc.pth', map_location=device)

# 用ypwhs训练完的模型('ctc.pth')进行预测
# 自己设备不行就没有训练
model.eval()
image, target, input_length, label_length = dataset[0]
# to_pil_image把tensor转成PIL image类型
pic = to_pil_image(image)
print('true:', decode_target(target))
output = model(image.unsqueeze(0))
output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
pred_num = decode(output_argmax[0])
print('pred:', pred_num)

pic.show()
# pic.save(pred_num+'.jpg')
