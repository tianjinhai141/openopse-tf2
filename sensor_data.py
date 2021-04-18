"""
Created on Mon Nov 12 15:57:59 2018

@author:
"""

import pandas as pd
import numpy as np
import tensorflow as tf

SENSOR_PROP_DATA_FILE_PATH = '../data/train_data/2018-11-29/15-49-52/sensor/2018-11-29.csv'
SENSOR_PROP_FPS = 4
SENSOR_PROP_FRAME_PER_IMAGE = 8


def create_tsmeta_file():
    '''
    Create the timestamp meta file with the same name as data file, but different suffix '.meta'
    The number of rows contained in meta file must equal to the number of image multiply SENSOR_PROP_FRAME_PER_IMAGE
    :return: None
    '''
    input_file = pd.read_csv(SENSOR_PROP_DATA_FILE_PATH)

    input_data = np.array(input_file.iloc[2:].values)

    millsec_interval = 1000 // SENSOR_PROP_FPS
    dataset_buf = []
    num_same_sec = 0
    tmp = None
    for row in input_data:
        ts_sec = row[0].split('|')[0]
        if ts_sec == tmp:
            num_same_sec = num_same_sec + 1
        else:
            num_same_sec = 0
        ts_mill_sec = ts_sec + ':' + str(min(millsec_interval * num_same_sec, 999)).zfill(3)
        ts_format = ts_mill_sec.replace(':', '-')
        dataset_buf.append(ts_format)
        tmp = ts_sec

    enum_over = len(dataset_buf) % SENSOR_PROP_FRAME_PER_IMAGE
    tsmeta_path = SENSOR_PROP_DATA_FILE_PATH.replace('csv', 'meta')
    np.savetxt(tsmeta_path, np.array(dataset_buf[:-enum_over]), fmt='%s')


def read_tsmeta():
    '''
    Read all timestamp from file with suffix '.meta'
    :return:
        ndarray with dtype=str
    '''
    tsmeta_path = SENSOR_PROP_DATA_FILE_PATH.replace('csv', 'meta')
    tsmeta_data = np.loadtxt(tsmeta_path, dtype=str)
    return tsmeta_data


def next_batch_tsmeta(input_data, index, batch_size):
    '''
    Return the corresponding timestamp of the image.
    The timestamp of one image is the timestamp of the latest frame contained in that image.
    :param input_data:
    :param index:
    :param batch_size:
    :return:
        ndarray with shape as (batch_size,), dtype=str

    '''
    data_len = len(input_data)
    start_index = (index * batch_size * SENSOR_PROP_FRAME_PER_IMAGE) % data_len
    end_index = start_index + batch_size * SENSOR_PROP_FRAME_PER_IMAGE
    result = []
    if end_index > data_len:
        first_part = input_data[start_index:]
        second_part = input_data[0: end_index - data_len]
        result = np.concatenate((first_part, second_part), axis=0)
        # return np.array(result)
    else:
        result = input_data[start_index: end_index]
        # return np.array(result)
    # print(result)
    return np.array(result)[SENSOR_PROP_FRAME_PER_IMAGE - 1::SENSOR_PROP_FRAME_PER_IMAGE]


def read_sendata():
    '''
        Read all image data stored in file.
        Each image includes multiple frames specified by SENSOR_PROP_FRAME_PER_IMAGE.
    :return:
        ndarray with shape as (None, SENSOR_PROP_FRAME_PER_IMAGE, 24, 32, 1)
    '''
    input_file = pd.read_csv(SENSOR_PROP_DATA_FILE_PATH)
    input_data = np.array(input_file.values)

    all_data = []
    image_data = []
    index = 0
    for line_str in input_data:
        line_array = line_str[0].split('|')
        row_value = line_array[2:770]
        line_data = np.array(row_value)
        frame_data = line_data.astype(np.float32).reshape(24, 32, 1)
        image_data.append(frame_data)

        index = index + 1
        if index % SENSOR_PROP_FRAME_PER_IMAGE == 0:
            all_data.append(image_data)
            image_data = []

    datasets = np.array(all_data)
    return datasets


def next_batch_sendata(input_data, index, batch_size):
    '''

    :param input_data: ndarray
    :param index: start from zero
    :param batch_size:
    :return:
        ndarray with shape as (batch_size, SENSOR_PROP_FRAME_PER_IMAGE, 24, 32, 1)
        subset of the input_data with index [index * batch_size, index * batch_size + batch_size)
        if reach the end of the input_data, it will go to head of the input_data
    '''
    data_len = len(input_data)
    start_index = (index * batch_size) % data_len
    end_index = start_index + batch_size
    result = []
    if end_index > data_len:
        first_part = input_data[start_index:]
        second_part = input_data[0: end_index - data_len]
        result = np.concatenate((first_part, second_part), axis=0)
        return np.array(result)
    else:
        result = input_data[start_index: end_index]
        return np.array(result)


if __name__ == '__main__':
    # create_tsmeta_file()

    tsmeta_data = read_tsmeta()
    tsmeta_batch = next_batch_tsmeta(tsmeta_data, 373, 3)
    print('tsmeta_batch:\n', tsmeta_batch)

    sendata = read_sendata()
    sendata_batch = next_batch_sendata(sendata, 373, 3)
    print('sendata_batch:\n', sendata_batch)

    print(len(tsmeta_data), len(sendata))
