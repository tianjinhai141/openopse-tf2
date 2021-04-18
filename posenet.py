# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tf_slim as slim
import sensor_data as sd
import heatMat_input as hmi
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.misc as misc
from PIL import Image
import time
import os
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# 若多个GPU的话
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指明了两个GPU ID号，注意这里不区分双引号和单引号
INPUT_THERMO_FRAME_HEIGHT = 24
INPUT_THERMO_FRAME_WIDTH = 32
INPUT_THERMO_FRAME_CHANNEL = 1
INPUT_THERMO_FRAMES_PER_IMAGE = 8

# OUTPUT_HEATMAP_HEIGHT = 184
# OUTPUT_HEATMAP_WIDTH = 216
OUTPUT_HEATMAP_HEIGHT = 96
OUTPUT_HEATMAP_WIDTH = 128
OUTPUT_HEATMAP_FRAMES = 1
OUTPUT_HEATMAP_CHANNEL = 1
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_1')
TRAINING_DATA_BATCH_SIZE = 256
TRAINING_STEPS = 32044

LEARNING_RATE_BASE = 0.07  # 1e-4
LEARNING_RATE_DECAY = 0.97
base_path = os.getcwd()
SENSOR_PROP_DATA_FILE_PATH =os.path.join( base_path,'sensor','2018-11-29.meta')
INdex_PROP_DATA_FILE_PATH =os.path.join(base_path,'index.meta')
# checkpoint_save_path = os.path.join(base_path, 'checkpoint', 'posenet_checkpoint.ckpt')

SENSOR_PROP_FRAME_PER_IMAGE = 8


def prelu(input, name=''):
    alphas = tf.compat.v1.get_variable(name=name + 'prelu_alphas',
                             initializer=tf.constant(0.25, dtype=tf.float32, shape=[input.get_shape()[-1]]))
    pos = tf.nn.relu(input)
    neg = alphas * (input - abs(input)) * 0.5
    return pos + neg


def inference(input_image):
    print('Input image:', input_image)
    with tf.compat.v1.variable_scope('posenet'):
        with slim.arg_scope([slim.conv3d],
                            padding='SAME',
                            activation_fn=tf.nn.leaky_relu,
                            kernel_size=[2, 3, 3],
                            stride=[1, 1, 1],
                            normalizer_fn=slim.batch_norm
                            ):
            net = slim.conv3d(input_image, 64, stride=[2, 2, 2], scope='conv1')
            net = slim.conv3d(net, 32, stride=[2, 1, 1], scope='conv2')
            net = slim.conv3d(net, 16, stride=[2, 1, 1], scope='conv3')
            net = slim.conv3d(net, 16, scope='conv4')
            net = slim.conv3d(net, 16, scope='conv5')
            net = slim.conv3d(net, 64, scope='conv6')
            print('Conv Net:', net)

        with slim.arg_scope([slim.conv3d_transpose],
                            padding='SAME',
                            activation_fn=tf.nn.leaky_relu,
                            kernel_size=[1, 3, 3],
                            stride=[1, 2, 2]
                            ):
            net = slim.conv3d_transpose(net, 32, scope='deconv1')
            net = slim.conv3d_transpose(net, 16, scope='deconv2')

            net = slim.conv3d_transpose(net, 1, scope='deconv3')
            print('Deconv Net:', net)

            return net



def read_tsmeta():
    '''
    Read all timestamp from file with suffix '.meta'
    :return:
        ndarray with dtype=str
    '''
    tsmeta_path = SENSOR_PROP_DATA_FILE_PATH.replace('csv', 'meta')
    tsmeta_data = np.loadtxt(tsmeta_path, dtype=str)
    return tsmeta_data


total_npz = np.load(os.path.join(base_path, 'hhh', 'total.npz'))['arr_0']
base_heatMat_index = INdex_PROP_DATA_FILE_PATH
npz_index = np.loadtxt(base_heatMat_index, dtype=str)
def get_lable_by_name(lable_names):
    valarr = []
    for row in npz_index:
        subrow = row[:-4].split('-')
        value = int(subrow[0]) * 60 * 60 + int(subrow[1]) * 60 + int(subrow[2]) + int(subrow[3]) * 0.001
        valarr.append(value)
    lable_list = []
    for row in lable_names:
        subrow = row.split('-')
        value = int(subrow[0]) * 60 * 60 + int(subrow[1]) * 60 + int(subrow[2]) + int(subrow[3]) * 0.001
        distance = np.abs(np.subtract(valarr, value))
        item = np.array(Image.fromarray(total_npz[np.argmin(distance)]).resize((128, 96)))
        # item = (item - np.min(item)) / (np.max(item) - np.min(item))
        lable_list.append(item)
    lable_list = np.array(lable_list)
    return lable_list


def train_model():
    input_image = tf.compat.v1.placeholder(dtype=tf.float32,
                                 shape=[None, INPUT_THERMO_FRAMES_PER_IMAGE, INPUT_THERMO_FRAME_HEIGHT,
                                        INPUT_THERMO_FRAME_WIDTH, INPUT_THERMO_FRAME_CHANNEL], name='input_image')
    ground_truth_heatmap = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, OUTPUT_HEATMAP_HEIGHT,
                                                                   OUTPUT_HEATMAP_WIDTH], name='ground_truth_heatmap')
    output_tensor_heatMat = inference(input_image)

    output_tensor_reshape = tf.reshape(output_tensor_heatMat, shape=[-1, OUTPUT_HEATMAP_HEIGHT, OUTPUT_HEATMAP_WIDTH])
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE, global_step, 100, LEARNING_RATE_DECAY)
    total_lable = get_lable_by_name(read_tsmeta()[::SENSOR_PROP_FRAME_PER_IMAGE])
    with tf.compat.v1.variable_scope('posenet'):
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ground_truth_heatmap, logits=output_tensor_reshape))

        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    total_loss = []
    total_acc = []
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        tf.compat.v1.local_variables_initializer().run()

        variable_to_store = slim.get_variables_to_restore(include=['posenet'], exclude=['groundtruth'])
        saver = tf.compat.v1.train.Saver(variable_to_store)
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        all_images = sd.read_sendata()
        all_tsmeta = sd.read_tsmeta()
        total_truth = []
        for j in range(TRAINING_STEPS):
            t_start = time.time()
            image_batch = sd.next_batch_sendata(all_images, j, TRAINING_DATA_BATCH_SIZE)

            tsmeta_batch = sd.next_batch_tsmeta(all_tsmeta, j, TRAINING_DATA_BATCH_SIZE)
            output_batch = get_lable_by_name(tsmeta_batch)
            ground_truth_format_value = np.where(output_batch > 0.9, 0.0, 1.0)
            train_reslt, loss, predict_value = sess.run([train_op, cross_entropy, output_tensor_reshape],
                                                        feed_dict={input_image: image_batch,
                                                                   ground_truth_heatmap: ground_truth_format_value})
            dis = np.equal(ground_truth_format_value, np.where(predict_value > 0.9, 1.0, 0.0)).sum()

            # loss_distribute_value = sess.run(loss_distribute, feed_dict={input_image: image_batch, ground_truth_heatmap: output_batch})
            # print('loss_distribute_value:\n', loss_distribute_value)
            # output_sigmoid_heatMat_value = sess.run(output_sigmoid_heatMat, feed_dict={input_image: image_batch, ground_truth_heatmap: output_batch})
            # print('output_sigmoid_heatMat_value:\n', output_sigmoid_heatMat_value)
            # ground_truth_heatmap_clip_value = sess.run(ground_truth_heatmap_clip, feed_dict={input_image: image_batch,
            #                                                                            ground_truth_heatmap: output_batch})
            # print('ground_truth_heatmap_clip_value:\n', ground_truth_heatmap_clip_value)
            v_shape = predict_value.shape
            acc_rate = dis / (v_shape[0] * v_shape[1] * v_shape[2])
            total_loss.append(loss)
            total_acc.append(acc_rate)
            if j % 1 == 0:
                normal_predict_result = (predict_value - np.min(predict_value)) / (
                        np.max(predict_value) - np.min(predict_value))
                show_figure(ground_truth_format_value,
                            predict_value,
                            str(j + 1) + '.jpg')
                print('Train epoch=========> %d,Time===========>%fs, Loss=========> %f ,Train_Acc========> %f' % (
                    j, time.time() - t_start, loss, acc_rate))
                if loss < 0.14 and acc_rate > 0.93:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, 'ckpt_' + str(loss))[:4] + '_' + str(acc_rate)[:4],
                               global_step=j)

                    # np.where(predict_value < 0.9, 0, predict_value)


        show_acc_and_loss(total_loss, total_acc)


def show_figure(ground_truth, predict_result, name):
    bins = np.linspace(0.0, 1.0, 50)
    fig = plt.figure(dpi=100)
    fig.suptitle('step=' + name[:-4])
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('lable_confidence')
    plt.sca(ax1)
    plt.xlim(-0.1, 1.1)
    plt.hist(ground_truth[0, :, :].flatten(), align='mid')

    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('predict_confidence')
    plt.sca(ax3)
    plt.xlim(-0.1, 1.1)
    sigpredict = 1 / (1 + np.exp(-predict_result[0, :, :]))
    # 将有无关键点的概率拉开
    s = np.where(sigpredict < 0.55, 0, sigpredict+0.1)
    s = np.where(s > 0.9, 1, s)
    # normal_predict_result = (predict_result-np.min(predict_result))/(np.max(predict_result)-np.min(predict_result))
    plt.hist(s.flatten(), align='mid')
    # plt.show()

    ax2 = plt.subplot(2, 2, 2)
    plt.sca(ax2)
    ax2.set_title('lable_people')
    plt.xticks([0, 50, 100, 150])
    plt.imshow(ground_truth[0, :, :])
    plt.colorbar()

    ax4 = plt.subplot(2, 2, 4)
    plt.sca(ax4)
    ax4.set_title('predict_people')
    plt.xticks([0, 50, 100, 150])
    predict_result1 = np.where(predict_result < 0.9, 0, predict_result)
    plt.imshow(predict_result1[0, :, :])
    plt.colorbar()
    # plt.show(block=False)

    path = os.path.join(os.getcwd(), 'img_2', name)
    if not os.path.exists(path):
        plt.savefig(path)
    # plt.pause(1)
    plt.close()


def show_acc_and_loss(losses, acces):
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(acces, label='Training Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'img_loss_acc', 'loss_acc.jpg'))


if __name__ == '__main__':
    print('Start train...')
    train_model()
    # print('--------------------------没有加归一化，没有关键点的npz--------------------------------')
    # print('*********************************************')
    # print('npz原数组：')
    # print(get_lable_by_name(['15-59-47-807']))
    # j = np.where(get_lable_by_name(['15-59-47-807']) > 0.9, 0, 1)
    # print('不归一化np.where变换之后npz数组',j)
    # print('总像素：', j.size, '，关键点的个数：', np.count_nonzero(j))
    # print('------------------------没有加归一化，有关键点的npz---------------------------------')
    # print('npz原数组：')
    # print(get_lable_by_name(['15-50-23-091']))
    # i = np.where(get_lable_by_name(['15-50-23-091']) > 0.9, 0, 1)
    # print('不归一化np.where变换之后npz数组',i)
    # print('总像素：',i.size,'，关键点的个数：',  np.count_nonzero(i))
    #
    # print('------------------------加入归一化后，没有关键点的npz----------------------------------')
    # print('npz原数组：')
    # nopeople_old_npz = get_lable_by_name(['15-59-47-807'])
    # print(nopeople_old_npz)
    # normallization_no_people_npz = (nopeople_old_npz - np.min(nopeople_old_npz)) / (np.max(nopeople_old_npz) - np.min(nopeople_old_npz))
    # j = np.where(normallization_no_people_npz> 0.9, 0, 1)
    # print('np.where变换之后npz数组', j)
    # print('总像素：', j.size, '，关键点的个数：', np.count_nonzero(j))
    # print('----------------------------加入归一化后，有关键点的npz------------------------------')
    # print('npz原数组：')
    # have_people_npz = get_lable_by_name(['15-50-23-091'])
    # print(have_people_npz)
    # normallization_have_people_npz = (have_people_npz - np.min(have_people_npz)) / (np.max(have_people_npz) - np.min(have_people_npz))
    # i = np.where(normallization_have_people_npz> 0.9, 0, 1)
    # print('np.where变换之后npz数组', i)
    # print('总像素：', i.size, '，关键点的个数：', np.count_nonzero(i))

    print('End train')
