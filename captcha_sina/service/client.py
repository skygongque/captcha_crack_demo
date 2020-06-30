import requests
import base64
import random

def get_captcha():
    r = int(random.random()*100000000)
    params = {
        'r': str(r),
        's': '0',
    }
    response = requests.get('https://login.sina.com.cn/cgi/pin.php',params=params)
    if response.status_code ==200:
        return response.content


payload = {
    'img':base64.b64encode(get_captcha()) 
}
response = requests.post('http://127.0.0.1:5000/sina',data=payload)
print(response.text)
