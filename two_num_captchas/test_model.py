import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
import random
import numpy as np

import string

characters =  string.digits
width, height, n_len, n_classes = 80, 40, 4, len(characters)
label_length = 2
# n_input_length = 12
print(characters, width, height, n_classes)

class TwoNumCaptchaDataset(Dataset):
    """ 两位数字验证码的图片 """
    def __init__(self, characters, length, width, height, label_length):
        super(TwoNumCaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        # self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def encode_label(self,target_str):
        # 编码
        target = []
        for char in target_str:
            vec = [0.0] * len(self.characters)
            vec[self.characters.find(char)] = 1.0
            target += vec
        return target

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        image = to_tensor(self.generator.generate_image(random_str))
        # 进行one_hot编码
        target = torch.tensor(self.encode_label(random_str)) 
        # target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
        return image, target


class CNN(nn.Module):
    def __init__(self,num_class=10, num_char=2):
        super(CNN,self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
                # b*3*80*40
                nn.Conv2d(3, 16, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # b*16*40*20
                nn.Conv2d(16, 64, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # b*64*20*10
                )
        self.fc = nn.Linear(64*10*20, self.num_class*self.num_char)
        # temp = torch.randn(2,3,40,80)
        # out = self.conv(temp)
        # print(out.shape)


    def forward(self, x):
        x = self.conv(x)
        # -1 适应后面的size
        x = x.view(-1,64*10*20)
        x = self.fc(x)
        return x

dataset = TwoNumCaptchaDataset(characters=characters,length=1,width=width,height=height,label_length=2)
image , target = dataset[0]
to_pil_image(image).show()

device = torch.device('cpu')
cnn = CNN()
cnn.load_state_dict(torch.load('cnn_6_27.pth',map_location=device))
cnn.eval()

# 用unsqueeze 增加一个维度
image = image.unsqueeze(0)

# 送入模型中进行预测
output = cnn(image)

# print(output.shape)
# 转成[2,10]
output = output.view(-1, 10)
# 计算 nn.functional.softmax转成 0-1 概率
output = nn.functional.softmax(output, dim=1)
output = torch.argmax(output, dim=1)
output = output.view(-1, 2)[0]
predict =''.join([characters[i] for i in output.cpu().numpy()])
print(predict)
