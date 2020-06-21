# from captcha.image import ImageCaptcha

# pic_generator = ImageCaptcha(180,50)
# pic = pic_generator.generate_image('1234')
from torchvision.transforms.functional import to_tensor, to_pil_image

from PIL import Image
pic = Image.open('101.jpg')
# print(pic.show())
# 图片转向量
pic_tensor = to_tensor(pic)



