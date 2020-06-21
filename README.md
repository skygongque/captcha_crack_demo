# 利用机器学习破解验证码尝试

> [学习的仓库链接](https://github.com/ypwhs/captcha_break)  
> 作者：ypwhs

导入ypwhs大佬训练的模型，试了一些对captcha生成的验码确实可以几乎100%正确识别。  

可以识别的验证码示例  
![可以识别的验证码](.\cracked_captchas\MBLG.jpg)

尝试换了一种验证码用原模型识别基本全错。
换的一种验证码示例  
![换的一种验证码示例](.\other_captchas\101.jpg)

得出结论要**对每种类型的验证码单独训练**  

# 导入模型中踩的坑总结  
torch加载模型前**要定义好相同的模型**（参数未优化），加载模型相当于加载优化后的参数  
加载模型 映射到cpu-only的设备  
```
device = torch.device('cpu')
model = torch.load('ctc.pth', map_location=device)
```

更多保存和加载模型的细节参见[官网SAVING AND LOADING MODELS](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-multiple-models-in-one-file)

# 更多

不能使用gpu(cuda)计算根本做不了图形识别  




