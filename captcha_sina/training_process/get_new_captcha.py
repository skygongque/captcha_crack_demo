import random
import requests
from PIL import Image


def get_pin():
    """ 保存验证码图片 """
    r = int(random.random()*100000000)
    # print(r)
    params = {
        'r': str(r),
        's': '0',
    }
    response = requests.get('https://login.sina.com.cn/cgi/pin.php',params=params)
    if response.status_code ==200:
        with open('pin_img.jpg','wb') as f:
            f.write(response.content)
            f.close()
            print('had saved Captcha.')
        # Image好像会调用默认的图片查看器打开图片
        I = Image.open('pin_img.jpg')
        # I.show()

if __name__ == "__main__":
    get_pin()
        