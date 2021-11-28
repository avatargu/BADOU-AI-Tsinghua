"""
分别使用"OpenCV、Matplotlib和PIL"实现彩色图像的"灰度化和二值化"
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

"""
OpenCV与彩色图像的灰度化
"""
# OpenCV读取彩色图像
img = cv2.imread("img/lenna_512x512.png")  # OpenCV读取三通道彩色图像，返回numpy三维数组
# print(img)  # 0~255
height, width, channel = img.shape  # 获取图像的高度、宽度、通道数
data_type = img.dtype  # 获取图像的数据类型
cv2.imshow("img", img)  # 显示三通道彩色图像
cv2.waitKey(1000)  # 等待1秒钟

# 手动灰度化彩色图像
img_gray = np.zeros([height, width], data_type)  # 创建numpy二维数组，就是创建单通道灰度图像
for h in range(height):  # 遍历高度坐标
    for w in range(width):  # 遍历宽度坐标
        bgr = img[h, w]  # 获取对应高度和宽度坐标的像素值（注意：对于三通道彩色图像，OpenCV读取的通道顺序为bgr）
        img_gray[h, w] = int(bgr[0] * 0.11 + bgr[1] * 0.59 + bgr[2] * 0.3)  # BGR转化为GRAY
# print(img_gray)  # 0~255
cv2.imshow("img_gray_2", img_gray)  # 显示单通道灰度图像
cv2.waitKey(1000)  # 等待1秒钟

# OpenCV灰度化彩色图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # OpenCV灰度化彩色图像
# print(img_gray)  # 0~255
# print(img_gray.shape)  # height, width
cv2.imshow("img_gray_3", img_gray)  # 显示单通道灰度图像
cv2.waitKey(1000)  # 等待1秒钟

# OpenCV以灰度模式读取彩色图像
img_gray = cv2.imread("img/lenna_512x512.png", 0)  # OpenCV读取三通道彩色图像，返回numpy二维数组
# print(img_gray)  # 0~255
# print(img_gray.shape)  # height, width
cv2.imshow("img_gray_4", img_gray)  # 显示单通道灰度图像
cv2.waitKey(1000)  # 等待1秒钟

"""
OpenCV与灰度图像的二值化
"""
# OpenCV二值化灰度图像
thresh, img_binary = cv2.threshold(
    src=img_gray,  # 灰度图像
    thresh=127,  # 阈值
    maxval=255,  # 最大值
    type=cv2.THRESH_BINARY  # 如果像素值小于阈值，则将其设置为0，否则将其设置为最大值
)
# print(img_binary)  # 0~255
# print("ret:", thresh)  # 127.0
# print(img_binary.shape)  # height, width
cv2.imshow("img_binary", img_binary)  # 显示单通道二值图像
cv2.waitKey(1000)  # 等待1秒钟

"""
Matplotlib与彩色图像的灰度化
"""
# Matplotlib读取彩色图像
img = plt.imread("img/lenna_512x512.png")  # Matplotlib读取三通道彩色图像，返回numpy三维数组
# print(img)  # 0~1
plt.subplot(131)  # 选中子图
plt.imshow(img)  # 绘制三通道彩色图像

# skimage灰度化彩色图像
img_gray = rgb2gray(img)  # skimage灰度化彩色图像
# print(img_gray)  # 0~1
height, width = img_gray.shape  # 获取图像的高度、宽度
data_type = img_gray.dtype  # 获取图像的数据类型
plt.subplot(132)  # 选中子图
plt.imshow(img_gray, cmap='gray')  # 绘制单通道灰度图像

"""
Matplotlib与灰度图像的二值化
"""
# 手动二值化灰度图像
# img_binary = np.zeros([height, width], data_type)  # 创建numpy二维数组，就是创建单通道二值图像
# rows, cols = img_gray.shape
# for row in range(rows):
#     for col in range(cols):
#         if img_gray[row, col] <= 0.5:
#             img_binary[row, col] = 0
#         else:
#             img_binary[row, col] = 1
img_binary = np.where(img_gray >= 0.5, 1, 0)  # numpy二值化灰度图像（一行代替上面八行）
# print(img_binary)  # 0或1
plt.subplot(133)  # 选中子图
plt.imshow(img_binary, cmap='gray')  # 绘制单通道灰度图像
plt.show()

"""
PIL与彩色图像的灰度化
"""
# PIL灰度化彩色图像
img = Image.open('img/lenna_512x512.png')    # PIL打开三通道彩色图像
"""
When translating a color image to greyscale (mode "L"),
the library uses the ITU-R 601-2 luma transform::
    L = R * 299/1000 + G * 587/1000 + B * 114/1000
"""
img_gray = img.convert('L')  # PIL灰度化彩色图像
img_gray.save('img/lenna_512x512_gray.jpg')  # PIL保存灰度图像

"""
PIL与灰度图像的二值化
"""
# PIL二值化灰度图像
threshold = 128  # 阈值
table = []  # 二值列表
for i in range(256):  # 0~255
    if i < threshold:  # 小于阈值
        table.append(0)  # 0
    else:  # 大于阈值
        table.append(1)  # 1
img_binary = img_gray.point(table, '1')  # PIL二值化灰度图像
img_binary.save('img/lenna_512x512_binary.png')  # PIL保存二值图像
