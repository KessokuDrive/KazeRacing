import requests
import time
import aiofiles
import numpy as np
import io
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg

import cv2  # OpenCV
import pillow_heif
from io import BytesIO
from PIL import Image

import json

from pillow_heif._libheif_ctx import LibHeifCtxWrite
from pillow_heif.constants import HeifCompressionFormat
camera = CSICamera(width=640, height=480, capture_fps=24)


t_all = []

url = 'http://192.168.31.7:8000/uploadfile'
headers = {
    'filename': 'output.avif',
    'Content-Type':'image/avif'
    }
#Capture Image
raw_image = camera.read()
heif_file = pillow_heif.from_bytes(
    mode="BGR",
    size=(raw_image.shape[1], raw_image.shape[0]),
    data=bytes(raw_image)
)
print("Saving")
#img = LibHeifCtxWrite(compression_format=HeifCompressionFormat.AV1)
#img.set_encoder_parameters(None,75)
#print(len(img))
heif_file.save("output.avif",format="AVIF", quality=75)
print("Reading")



# start = time.time()
# r = requests.post(url=url, data=image, headers=headers)
# end = time.time() - start
import aiohttp
import asyncio


globalTime = time.time()
count = 0
async def main(count):
    async with aiohttp.ClientSession() as session:
        async def file_sender(file_name=None):
            async with aiofiles.open(file_name, 'rb') as f:
                chunk = await f.read(64*1024)
                while chunk:
                    yield chunk
                    chunk = await f.read(64*1024)
        """ async with session.post(url,
                                data=file_sender(file_name='output.avif'),headers={
                                    'filename': 'output_'+str(count)+'.avif',
                                    'Content-Type':'image/avif'
                                }) as response:"""
        async with session.post(url,
                    data='',headers={
                        'filename': 'output_1.avif',
                        'Content-Type':'image/avif'
                    }) as response:
            return response.content

while True:
    check = time.time()
    count+=1
    if(int((check - globalTime)%60)>=10):
        break
    t1 = time.time()
    print(asyncio.get_event_loop().run_until_complete(main(count)))
    t2 = time.time()
    t_all.append(t2 - t1)
    


""" 
globalTime = time.time()
count = 0
s = requests.Session()
while True:
    check = time.time()
    if(int((check - globalTime)%60)>=10):
        break
    t1 = time.time()
    s.post(url=url, data=image, headers=headers)
    count+=1
    t2 = time.time()
    t_all.append(t2 - t1)"""
    

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))

print("count:",count)
#print(r.text)