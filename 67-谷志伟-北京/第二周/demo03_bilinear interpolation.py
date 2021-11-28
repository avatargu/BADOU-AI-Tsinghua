"""
双线性插值(bilinear interpolation)
"""
import numpy as np
import cv2


def bilinear_interpolation(src_img, dst_dim):
    """
    双线性插值：将源图像的宽度和高度分别缩放到指定值
    :param src_img: 数组类型，源图像
    :param dst_dim: 元组类型，目标图像分辨率：宽度（像素） x 高度（像素）
    :return: 数组类型，目标图像
    """
    # 获取源图像的高度、宽度、通道数
    src_height, src_width, num_channels = src_img.shape
    print("src_height, src_width, num_channels =", src_height, src_width, num_channels)

    # 获取目标图像的高度、宽度（注意：图像分辨率表示为宽度 x 高度）
    dst_height, dst_width = dst_dim[1], dst_dim[0]
    print("dst_height, dst_width =", dst_height, dst_width)

    # 提高性能
    if src_height == dst_height and src_width == dst_width:
        return src_img.copy()  # 如果图像没有缩放，直接返回源图像的拷贝

    # 初始化目标图像
    dst_img = np.zeros(
        (dst_height, dst_width, num_channels),  # 目标图像的高度、宽度、通道数
        dtype=np.uint8  # 0~255
    )

    # 计算图像缩放比例
    scale_height, scale_width = float(dst_height / src_height), float(dst_width / src_width)

    # 缩放源图像，得到目标图像
    for channel in range(num_channels):  # 遍历目标图像的三个通道
        for dst_h in range(dst_height):  # 遍历目标图像的高度坐标
            for dst_w in range(dst_width):  # 遍历目标图像的宽度坐标
                """
                0 1 2 3 4      ——> 图像坐标从0开始导致源图像和目标图像的几何左上角重合 
                |p|p|p|p|p|
                 0 1 2 3 4      ——> 图像坐标从0.5开始使得源图像和目标图像的几何中心重合
                """
                src_h = (dst_h + 0.5) / scale_height - 0.5  # 获取目标图像高度坐标对应的源图像高度坐标
                src_w = (dst_w + 0.5) / scale_width - 0.5  # 获取目标图像宽度坐标对应的源图像宽度坐标

                """
                组合得到左上角、右上角、左下角、右下角四个像素的坐标
                """
                src_h_up = int(np.floor(src_h))  # 上边最近邻两个像素的高度坐标
                src_h_down = min(src_h_up + 1, src_height - 1)  # 下边最近邻两个像素的高度坐标（"src_height-1"是为了防止高度坐标越界）
                src_w_left = int(np.floor(src_w))  # 左边最近邻两个像素的宽度坐标
                src_w_right = min(src_w_left + 1, src_width - 1)  # 右边最近邻两个像素的宽度坐标（"src_width-1"是为了防止宽度坐标越界）

                """
                单线性插值公式推导：
                    (y - y0) / (y1 - y0) = (x - x0) / (x1 - x0) 
                    => 
                    y = (x1 - x) / (x1 - x0) * y0 + (x - x0) / (x1 - x0) * y1
                    ="相邻像素宽度坐标差值为1，即x1 -x0"> 
                    y = (x1 - x) * y0 + (x - x0) * y1
                """
                # 第一次单线性插值
                unilinear_interpolation_1 = (src_w_right - src_w) * src_img[src_h_up, src_w_left, channel] + (src_w - src_w_left) * src_img[src_h_up, src_w_right, channel]
                # 第二次单线性插值
                unilinear_interpolation_2 = (src_w_right - src_w) * src_img[src_h_down, src_w_left, channel] + (src_w - src_w_left) * src_img[src_h_down, src_w_right, channel]
                # 第三次单线性插值
                dst_img[dst_h, dst_w, channel] = int((src_h_down - src_h) * unilinear_interpolation_1 + (src_h - src_h_up) * unilinear_interpolation_2)

    # 返回目标图像
    return dst_img


if __name__ == '__main__':
    src_img = cv2.imread("img/lenna_500x350.jpg")  # 读取源图像(返回numpy数组)：宽度为500，高度为350
    dst_img = bilinear_interpolation(src_img, (800, 600))  # 缩放源图像：设置目标图像宽度为800，高度为600

    cv2.imshow("src_img", src_img)  # 显示源图像
    cv2.imshow("dst_img", dst_img)  # 显示目标图像
    cv2.waitKey(0)  # 无限等待任何键盘事件
