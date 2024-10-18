#This cod gets perspective transform of a image using 
#predefined parameters.

IMG_PATH = "E:\KazeRacing\BEV\Advanced-Lane-Line-Detection\\test_images\\f1_large_0948.png"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2
import numpy as np
import pickle as pickle
import glob
import settings

roi_points = np.array([[0, settings.ORIGINAL_SIZE[1] - 50],
                     [settings.ORIGINAL_SIZE[0], settings.ORIGINAL_SIZE[1] - 50],
                     [settings.ORIGINAL_SIZE[0]//2, settings.ORIGINAL_SIZE[1]//2+50]], np.int32)
print('roi_points: ', roi_points)
# Find the region of interest
roi = np.zeros((720, 1280), np.uint8)
print('roi: ', roi.shape)
cv2.fillPoly(roi, [roi_points], 1)

#--------  Apply Processing to example  ----------------
img = np.array(Image.open(IMG_PATH))

# convert to hls
img_hs1 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

# Canny
# Use the lightness layer to detect lines
low_thresh = 100
high_thresh = 150

# Lightness thresholding (returns less points than saturation thresholding, gives better representation of lane lines)
edges_lightness = cv2.Canny(img_hs1[:,:,1], high_thresh, low_thresh)

#Line Detection
lines = cv2.HoughLinesP(edges_lightness, 0.5, np.pi/180, 20, None, 180, 120)

Lhs = np.zeros((2, 2), dtype = np.float32)
Rhs = np.zeros((2, 1), dtype = np.float32)
x_max = 0
x_min = 2555
for line in lines:
    for x1, y1, x2, y2 in line:
        # Find the norm (the distances between the two points)
        normal = np.array([[-(y2-y1)], [x2-x1]], dtype = np.float32) # question about this implementation
        normal = normal / np.linalg.norm(normal)
        
        pt = np.array([[x1], [y1]], dtype = np.float32)
        
        outer = np.matmul(normal, normal.T)
        
        Lhs += outer
        Rhs += np.matmul(outer, pt) #use matmul for matrix multiply and not dot product

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness = 1)
        
        x_iter_max = max(x1, x2)
        x_iter_min = min(x1, x2)
        x_max = max(x_max, x_iter_max)
        x_min = min(x_min, x_iter_min)

width = x_max - x_min
print('width : ', width)

# Calculate Vanishing Point
vp = np.matmul(np.linalg.inv(Lhs), Rhs)

# Find the transform matrix
def find_pt_inline(p1, p2, y):
    """
    Here we use point-slope formula in order to find a point that is present on the line
    that passes through our vanishing point (vp). 
    input: points p1, p2, and y. They come is as tuples [x, y]
    We then use the point-slope formula: y - b = m(x - a)
    y: y-coordinate of desired point on the line
    x: x-coordinate of desired point on the line
    m: slope
    b: y-coordinate of p1
    a: x-coodrinate of p1
    x = p1x + (1/m)(y - p1y)
    """
    m_inv = (p2[0] - p1[0]) / float(p2[1] - p1[1])
    Δy = (y - p1[1])
    x = p1[0] + m_inv * Δy
    return [x, y]

top = vp[1] + 65
bot = settings.ORIGINAL_SIZE[1] -210

# Make a large width so that you can grab the lines on the challenge video
width = 500
center = vp[0]
p1 = [center - width/2, top]
p2 = [center + width/2, top]
p3 = find_pt_inline(p2, vp, bot)
p4 = find_pt_inline(p1, vp, bot)

D = [0, 0]
dst_pts = np.float32([[0, 0], [settings.UNWARPED_SIZE[0], 0],
                       [settings.UNWARPED_SIZE[0], settings.UNWARPED_SIZE[1]],
                       [0, settings.UNWARPED_SIZE[1]]])
A = p1[0][0]
src_pts = np.float32([[p1[0][0],p1[1][0]],
                      [p2[0][0],p2[1][0]],
                      [p3[0][0],p3[1]],
                      [p4[0][0],p4[1]]])

# H is the homography matrix
M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(np.array(Image.open(IMG_PATH)), M, settings.UNWARPED_SIZE)
cv2.imwrite('./warped.png', cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))