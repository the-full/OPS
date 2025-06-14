# coding=utf-8
"""Evaluate the attack success rates under 7 models including normally trained models and adversarially trained models"""

import argparse
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave
import pandas as pd
from nets import inception_v3, inception_resnet_v2


slim = tf.contrib.slim

checkpoint_path = './models'
model_checkpoint_map = {
    'inception_v3': os.path.join(checkpoint_path, 'inception_v3.ckpt'),
    'ens3_adv_inception_v3': os.path.join(checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(checkpoint_path, 'resnet_v2_101.ckpt'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='args')

    # 添加参数
    parser.add_argument('--label_path', type=str, default='../TransferAttack/new_data/labels.csv')
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--GPU_ID', type=str, default='0')

    # 解析参数
    args = parser.parse_args()
    return args


def load_labels(file_name):
    dev = pd.read_csv(file_name)
    # NOTE: In Admix, the label starts counting from 1, while in TransferAttack, 
    #       the label starts counting from 0. Therefore, +1 is needed here.
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] + 1 for i in range(len(dev))}
    return f2l


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB')
            image = imresize(image, (299, 299)).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    f2l = load_labels(args.label_path)
    if args.attack != 'none':
        if 'ops' in args.attack:
            input_dir = os.path.join('../TransferAttack/exp_data/exp_ops_overhead/', args.attack, 'resnet18')
        else:
            input_dir = os.path.join('../TransferAttack/exp_data/exp_method_compare/', args.attack, 'resnet18')
    else:
        input_dir = args.input_dir

    batch_shape = [50, 299, 299, 3]
    num_classes = 1001
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ens_adv_res_v2, end_points_ens_adv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

        pred_ens3_adv_v3 = tf.argmax(end_points_ens3_adv_v3['Predictions'], 1)
        pred_ens4_adv_v3 = tf.argmax(end_points_ens4_adv_v3['Predictions'], 1)
        pred_ens_adv_res_v2 = tf.argmax(end_points_ens_adv_res_v2['Predictions'], 1)

        s1 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            s2.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            s3.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])

            model_name = ['ens3_adv_inception_v3', 'ens4_adv_inception_v3', 'ens_adv_inception_resnet_v2']
            success_count = np.zeros(len(model_name))

            idx = 0
            for filenames, images in load_images(input_dir, batch_shape):
                idx += 1
                print("start the i={} eval".format(idx))
                ens3_adv_v3, ens4_adv_v3, ens_adv_res_v2 = sess.run(
                    (pred_ens3_adv_v3, pred_ens4_adv_v3, pred_ens_adv_res_v2), 
                    feed_dict={x_input: images}
                )

                for filename, l1, l2, l3 in zip(filenames, ens3_adv_v3, ens4_adv_v3, ens_adv_res_v2):
                    label = f2l[filename]
                    l = [l1, l2, l3]
                    for i in range(len(model_name)):
                        if l[i] != label:
                            success_count[i] += 1

            for i in range(len(model_name)):
                print("Attack Success Rate for {0} : {1:.1f}%".format(model_name[i], success_count[i] / 1000. * 100))
