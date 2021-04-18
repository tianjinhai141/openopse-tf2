"""
Created on Mon Nov 12 15:57:59 2018

@author:
"""
import skimage.transform
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.misc as misc

HEATMAT_META_FILE_PATH = '../data/train_data/2018-11-29/15-49-52/index.meta'
HEATMAT_DIR__PATH = '../data/train_data/2018-11-29/15-49-52/heatMat'

def read_heatMat(tsarr,file_names):
    '''
    Read the output heatMat based on the input timestamp
    :param tsarr:
    :return:
    '''
    file_names = np.loadtxt(HEATMAT_META_FILE_PATH, dtype=str)
    valarr = []
    for row in file_names:
        subrow = row[:-4].split('-')
        value = int(subrow[0]) * 60 * 60 + int(subrow[1]) * 60 + int(subrow[2]) + int(subrow[3]) * 0.001
        valarr.append(value)
    # print(valarr)

    matched_file_names = []
    for row in tsarr:
        subrow = row.split('-')
        value = int(subrow[0]) * 60 * 60 + int(subrow[1]) * 60 + int(subrow[2]) + int(subrow[3]) * 0.001
        # np.abs() 函数返回数字的绝对值。
        # np.subtract(x1,x2),实现x2-x1
        distance = np.abs(np.subtract(valarr, value))
        # np.argmin 检测每个轴上的最小值，并返回下标
        matched_file_names.append(file_names[np.argmin(distance)])

    heatMat_col = []
    for row in matched_file_names:
        # os.path.join()函数实现连接两个或更多的路径名组件
        # replace 替换
        heatMat_fname = os.path.join(HEATMAT_DIR__PATH, row.replace('jpg', 'npz'))
        # np.load（）读入np格式数据
        heatMat = np.load(heatMat_fname)
        normalize_col = []
        for j in range(19):
            # skimage.transform.resize(image, output_shape)
            # image: 需要改变尺寸的图片
            # output_shape: 新的图片尺寸
            heatMat_resize = skimage.transform.resize(heatMat['arr_0'][:, :, j], (96, 128))
            hm_normalize = (heatMat_resize - np.min(heatMat_resize)) / (np.max(heatMat_resize) - np.min(heatMat_resize))
            normalize_col.append(hm_normalize)
        # np.array 构造函数
        arr_normal = np.array(normalize_col).transpose((1, 2, 0))
        heatMat_col.append(arr_normal)

    return np.array(heatMat_col)


if __name__ == '__main__':
    tsarr = ['15-50-27-256']
    heatMat = read_heatMat(tsarr)[0, :, :, 17]
    print(heatMat.shape)
    print(np.sum(heatMat))
    print(np.max(heatMat))
    print(np.min(heatMat))
    print(heatMat)

    plt.imshow(heatMat)
    plt.colorbar()
    plt.show()
    # np.linspace 在指定的间隔内返回均匀间隔的数字
    bins = np.linspace(0.0, 1.0, 20)  # 0-1之间生成均匀间隔数字，20个样本
    # plt.hist 绘制直方图
    plt.hist(heatMat.flatten(), bins)
    plt.show()

