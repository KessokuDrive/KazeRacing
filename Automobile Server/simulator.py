import requests
import time
import numpy as np

url = 'http://192.168.31.7:8000'
file_path = './Image/Screenshot.png'
print('Uploading file...')
headers = {
    'filename': 'output.png',
    'Content-Type':'image/png'
    }

"""
t_all = []

globalTime = time.time()
count = 0
while True:
    check = time.time()
    if(int((check - globalTime)%60)>=10):
        break
    t1 = time.time()
    requests.post(url=url, data='image', headers=headers)
    count+=1
    t2 = time.time()
    t_all.append(t2 - t1)
    

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))

print("count:",count)

"""
import requests
import time

with open("ffxiv_20240918_213234_597.png", "rb") as f:
    data = f.read()
   
url = 'http://127.0.0.1:8000/uploadfile/'
headers = {'filename': 'output.png',
            'Content-Type':'image/png'
           }

start = time.time()
r = requests.post(url=url, data=data, headers=headers)
end = time.time() - start

print(f'Elapsed time is {end} seconds.', '\n')
print(r.json())