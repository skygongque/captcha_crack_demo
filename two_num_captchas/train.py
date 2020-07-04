import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
import random
import numpy as np

import string

characters = string.digits
width, height, n_len, n_classes = 80, 40, 4, len(characters)
label_length = 2
# n_input_length = 12
print(characters, width, height, n_classes)

max_epoch = 15


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

    def encode_label(self, target_str):
        target = []
        for char in target_str:
            vec = [0.0] * len(self.characters)
            vec[self.characters.find(char)] = 1.0
            target += vec
        return target

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        random_str = ''.join([random.choice(self.characters[1:])
                              for j in range(self.label_length)])
        image = to_tensor(self.generator.generate_image(random_str))
        # 进行one_hot编码
        target = torch.tensor(self.encode_label(random_str))
        # target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
        return image, target


class CNN(nn.Module):
    def __init__(self, num_class=10, num_char=2):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # b*3*80*40
            nn.Conv2d(3, 16, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 64, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64*10*20, self.num_class*self.num_char)
        # temp = torch.randn(2,3,40,80)
        # out = self.conv(temp)
        # print(out.shape)

    def forward(self, x):
        x = self.conv(x)
        # -1 适应后面的size
        x = x.view(-1, 64*10*20)
        x = self.fc(x)
        return x


def calculat_acc(output, target):
    """ 计算准确率 """
    output, target = output.view(-1, 10), target.view(-1, 10)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 2), target.view(-1, 2)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


batch_size = 128
train_set = TwoNumCaptchaDataset(
    characters=characters, length=batch_size*1, width=width, height=height, label_length=2)
# 改num_workers增加速度
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0)
test_set = TwoNumCaptchaDataset(
    characters=characters, length=300, width=width, height=height, label_length=2)
test_loader = DataLoader(test_set, batch_size=batch_size)

cnn = CNN()
if torch.cuda.is_available():
    cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
criterion = nn.MultiLabelSoftMarginLoss()
# criterion = nn.CrossEntropyLoss()

for epoch in range(max_epoch):
    cnn.train()
    for img, target in train_loader:
        if torch.cuda.is_available():
            img = img.cuda()
            target = target.cuda()
        # print('target_shape',target.shape)
        output = cnn(img)
        # print('out_shape',output.shape)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch:{epoch}  train_loss:{loss:.4f}')

    # test
    with torch.no_grad():
        # loss_history = []
        acc_history = []
        cnn.eval()
        for img, target in test_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            test_out = cnn(img)
            acc = calculat_acc(test_out, target)
            acc_history.append(float(acc))
            # loss_history.append(float(loss))
        print('test_acc: {:.4}'.format(
            # torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history))
        ))

# save model
torch.save(cnn.state_dict(), 'cnn_6_27.pth')
print('saved')


if __name__ == "__main__":
    pass
