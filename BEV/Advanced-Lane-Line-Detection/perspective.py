import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2
import numpy as np
import pickle as pickle
import os
import settings



def transform_image(img,filename):
    dst_pts = np.float32([[0, 0], [settings.UNWARPED_SIZE[0], 0],
                       [settings.UNWARPED_SIZE[0], settings.UNWARPED_SIZE[1]],
                       [0, settings.UNWARPED_SIZE[1]]])

    src_pts = np.float32([[287.64636,236.55095],
                      [987.64636,236.55095],
                      [1588.3408,420],
                      [-251.8982,420]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(np.array(Image.open(img)), M, settings.UNWARPED_SIZE)
    cv2.imwrite('output_images/'+filename+'_warped.png', cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    

def process_images(directory):
    # 确保输出文件夹存在
    output_folder = 'output_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # 构建完整的文件路径
            base_filename = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            # 调用transform_image方法处理图像
            transform_image(file_path, base_filename)
if __name__ == '__main__':
    # 指定你的图片目录
    image_directory = r"E:\KazeRacing\Recorded Track\f1_large_TapeA"
    process_images(image_directory)