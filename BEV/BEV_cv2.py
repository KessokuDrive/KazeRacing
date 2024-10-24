import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
img = cv2.imread('Standard.png') 
 
# Create a copy of the image
img_copy = np.copy(img)
 
"""# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the 
# transformation matrix
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
plt.imshow(img_copy)
plt.show()"""

IMAGE_H = 720
IMAGE_W = 1280

src = np.float32([[0, IMAGE_H], [1280, IMAGE_H], [0, 200], [IMAGE_W, 200]])
dst = np.float32([[500, IMAGE_H], [700, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

img = np.array(Image.open('E:\KazeRacing\BEV\Advanced-Lane-Line-Detection\\test_images\Curved.png')) # Read the test img

warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
plt.show()
cv2.imwrite('./BEV.png', cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR))