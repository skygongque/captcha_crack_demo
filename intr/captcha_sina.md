# sina 验证码识别

## 准确率99%+

## 数据集来源
[数据集来源](https://bbs.nightteam.cn/thread-470.htm)

## 训练方法参考  
[ctc_pytorch](https://github.com/ypwhs/captcha_break)  

## 笔记
1. Dataset, DataLoader的搭建自己的数据集
2. 更改 n_input_length 使得其符合模型
3. 使得符合ctc_loss要求 减少一个池化层
4. 模型部分未完全理解

