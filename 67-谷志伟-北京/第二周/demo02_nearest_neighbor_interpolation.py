"""
最近邻插值(nearest_neighbor interpolation)
"""
import numpy as np
import cv2


def nearest_neighbor(src_img, dst_dim):
    """
    最近邻插值：将源图像的宽度和高度分别缩放到指定值
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
    for dst_h in range(dst_height):  # 遍历目标图像的高度坐标
        for dst_w in range(dst_width):  # 遍历目标图像的宽度坐标
            src_h = int(dst_h / scale_height)  # 获取目标图像高度坐标对应的源图像高度坐标
            src_w = int(dst_w / scale_width)  # 获取目标图像宽度坐标对应的源图像宽度坐标
            dst_img[dst_h, dst_w] = src_img[src_h, src_w]  # 用源图像对应坐标的像素值(三通道)值给目标图像赋值

    # 返回目标图像
    return dst_img


if __name__ == "__main__":
    src_img = cv2.imread("img/lenna_500x350.jpg")  # 读取源图像(返回numpy数组)：宽度为500，高度为350
    dst_img = nearest_neighbor(src_img, (800, 600))  # 缩放源图像：设置目标图像宽度为800，高度为600

    cv2.imshow("src_img", src_img)  # 显示源图像
    cv2.imshow("dst_img", dst_img)  # 显示目标图像
    cv2.waitKey()  # 无限等待任何键盘事件
