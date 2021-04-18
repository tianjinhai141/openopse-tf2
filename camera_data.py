import cv2
import numpy as np
import matplotlib.pyplot as plot
import time
import pandas as pd
import os
import shutil
import pickle


def read_frame_to_image():
    # set root dir to store frame
    cam_date_dir = '../data/camera/' + get_date()
    # os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径
    if not os.path.exists(cam_date_dir):
        # os.makedirs() 方法用于递归创建目录。
        os.makedirs(cam_date_dir)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # set resolution & fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 432)# 视频流中帧的宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 368)# 视频流中帧的高度
    cap.set(cv2.CAP_PROP_FPS, 5)# 帧速率
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('camera setting: resolution=%s, fps=%d' % (resolution, fps))# 摄像机设置:分辨率= ，fps=

    sample_interval = 250 # in millseconds  样本间隔
    num_image_per_fold = (1000 // sample_interval) * 60 * 60
    max_round = 100
    is_stop = False
    for rnd in range(max_round):
        data_dir = cam_date_dir + '/' + get_time_stamp()
        # os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径
        if os.path.exists(data_dir):
            # shutil.rmtree 递归删除一个目录以及目录内的所有内容
            shutil.rmtree(data_dir)
        # os.mkdir(path) 创建目录，其参数path为要创建目录的路径。
        os.mkdir(data_dir)

        for i in range(num_image_per_fold):
            # get a frame
            #  从摄像头中读取画面，for表示循环读取画面，也就是一张一张图片形成了一个视频
            ret, frame = cap.read()
            if ret == True:
                # cv2.flip 图像翻转进行数据增强
                # 用法如下： 1表示水平翻转，0表示垂直翻转，-1表示水平垂直翻转

                # image = cv2.imread("girl.jpg")
                #  水平翻转
                # h_flip = cv2.flip(image, 1)
                # cv2.imwrite("girl-h.jpg", h_flip)
                #
                #  垂直翻转
                # v_flip = cv2.flip(image, 0)
                # cv2.imwrite("girl-v.jpg", v_flip)

                # 水平垂直翻转
                # hv_flip = cv2.flip(image, -1)
                # cv2.imwrite("girl-hv.jpg", hv_flip)
                flip_frame = cv2.flip(frame, 1)
                # 图片缩放 cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)
                # 用法： cv2.resize(输入图像, 输出图像, 输出图片尺寸, 沿x轴的缩放系数, 沿y轴的缩放系数, 插入方式)
                resize_frame = cv2.resize(flip_frame, (432, 368))
                # cv2.imread(filepath,flags)读入一副图片
                # filepath：要读入图片的完整路径
                # flags：读入图片的标志
                # cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
                # cv2.IMREAD_GRAYSCALE：读入灰度图片
                # cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道

                # cv2.imshow(wname,img)显示图像
                # 第一个参数是显示图像的窗口的名字，第二个参数是要显示的图像，窗口大小自动调整为图片大小
                cv2.imshow('capture', resize_frame)
                frame_path = data_dir + '/' + get_time_stamp_millsecond() + '.jpg'
                # cv2.imwrite(file，img，num)保存一个图像
                # 第一个参数是要保存的文件名，第二个参数是要保存的图像。可选的第三个参数，
                # 它针对特定的格式：对于JPEG，其表示的是图像的质量，用0 - 100的整数表示，默认95;
                # 对于png ,第三个参数表示的是压缩级别，默认为3
                cv2.imwrite(frame_path, frame)

            # time.sleep(1)
            # cv2.waitKey(delay=None) 函数的功能是不断刷新图像，频率为delay
            if cv2.waitKey(sample_interval) & 0xFF == ord('q'):
                is_stop = True
                break

        if is_stop:
            break
    # # 释放资源
    cap.release()
    # 关闭窗口
    cv2.destroyAllWindows()


def get_time_stamp():  # 获取时间戳
    ct = time.time()
    local_time = time.localtime(ct)
    time_stamp = time.strftime('%H-%M-%S', local_time)
    return time_stamp


def get_time_stamp_millsecond():
    ct = time.time()
    local_time = time.localtime(ct)
    # data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_head = time.strftime('%H-%M-%S', local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = '%s-%03d' % (data_head, data_secs)
    return time_stamp


def get_date():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime('%Y-%m-%d', local_time)
    return data_head




def read_frame_to_bin():
    # start = 100
    # b = np.reshape(np.arange(start, start + 432 * 368 * 3), (432, 368, 3))
    # bf = b.flatten(order='C')
    # print(b)
    # print(bf)
    #
    # start = 200
    # c = np.reshape(np.arange(start, start + 432 * 368 * 3), (432, 368, 3))
    # cf = c.flatten(order='C')
    #
    # start = 300
    # d = np.reshape(np.arange(start, start + 432 * 368 * 3), (432, 368, 3))
    # df = d.flatten(order='C')
    #
    # buf = []
    # buf.append(bf)
    # buf.append(cf)
    # buf.append(df)
    # fr_buf = np.array(buf)
    # print(buf)
    # print(fr_buf)
    #
    # ts_buf = np.reshape(np.arange(1000, 1000 + 3), (3, 1))
    # print(ts_buf)

    array_buf = np.reshape(np.arange(432 * 368 * 5000), (-1, 432 * 368))
    print('array_buf:\n', array_buf)
    # 将数组保存到文本文件
    np.savetxt('../data/camera/img_data_test.csv', array_buf, fmt='%d', delimiter='|', newline='\n')
    # np.loadtxt()用于从文本加载数据。
    # 文本文件中的每一行必须含有相同的数据。
    # 使用方法：loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None,
    #               converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    # fname要读取的文件、文件名、或生成器。
    # dtype数据类型，默认float。
    # comments注释。
    # delimiter分隔符，默认是空格。
    # skiprows跳过前几行读取，默认是0，必须是int整型。
    # usecols：要读取哪些列，0是第一列。例如，usecols = （1,4,5）将提取第2，第5和第6列。默认读取所有列。
    # unpack如果为True，将分列读取

    txt_buf = np.loadtxt('../data/camera/img_data_test.csv', dtype='int', delimiter='|')
    print('txt_buf:\n', txt_buf)
    print(txt_buf[2][432 * 368 - 2])

    # with open('../data/camera/img_data_test.txt', mode='w') as file:
    #     pickle.dump(fr_buf, file)

    # print(bf.size)
    # content = ''
    # for i in range(bf.size) :
    #     content = content + '|' + str(bf[i])
    #     # print('content: ' + content)
    #
    # content = content + '\n'
    # print('content: ' + content)

    # data = pd.DataFrame(fr_buf)
    # data.to_csv('../data/camera/img_data_test.csv', index=False, header=False)

    # with open('../data/camera/img_data_test.csv', mode='w') as file:
        # for i in range(fr_buf.size) :
        #     cnt = fr_buf[i]
    # np.savetxt('../data/camera/img_data_test.csv', fr_buf, fmt='%d', delimiter='|', newline='\n')

    # len = fr_buf.size
    # with open('../data/camera/img_data_test.txt', mode='w') as file:
    #     for i in range(len) :
    #         fr_item = fr_buf[i]
    #         ts_item = ts_buf[i]
    #         # line = ts_item + ' | ' + fr_item
    #         # np.savetxt(file, line)
    #         file.write(ts_item + ' | ')
    #         file.write(fr_item + ' | ')
    #
    #
    #     for (k,v) in buffer.items():
    #         # line = k + '-' + str(v)
    #         file.write(k + '-')
    #         np.savetxt(file, v)


if __name__ == '__main__':
    # read_frame_to_image()
    read_frame_to_bin()