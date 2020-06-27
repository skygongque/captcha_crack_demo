# 两位数字的验证码识别
## 准确率90%+

> 部分代码参考[JT623/Captcha](https://github.com/JT623/Captcha)

## 最简单的卷积神经网络(cnn)

```
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
                
                nn.Conv2d(16, 64, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                )
        self.fc = nn.Linear(64*10*20, self.num_class*self.num_char)
        # 一种测试模型经过部分运算后shape的方法
        # temp = torch.randn(2,3,40,80)
        # out = self.conv(temp)
        # print(out.shape)


    def forward(self, x):
        x = self.conv(x)
        # -1 适应后面的size
        x = x.view(-1,64*10*20)
        x = self.fc(x)
        return x
```

## 在云GPU上计算的结果
[Two_num_captcha.ipynb](two_num_captchas\Two_num_captcha.ipynb)  
[mistgpu注册](https://mistgpu.com/i/227504)   
