import os
import numpy as np

from model import FlowGAN
from utils import pp  # , visualize

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 256, "The size of image to use (will be center cropped). [256]")
flags.DEFINE_integer("input_width", 341, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 128, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("data_file", "./train_genlist_all_img_5frames.txt", "Name of dataset file")
flags.DEFINE_string("data_dir", "/scratch/pkwang/UCF101_frames_org2/ApplyEyeMakeup/", "Directory of data")
flags.DEFINE_string("data_dir_flow", "/scratch/pkwang/UCF101_opt_flows_org2/ApplyEyeMakeup/", "Directory of data")
flags.DEFINE_string("dataset_name", "Makeup5Frames", "Name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    run_config.gpu_options.allow_growth = True

    # decide which GPU's are visible
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with tf.Session(config=run_config) as sess:
        flowgan = FlowGAN(
            sess,
            data_file=FLAGS.data_file,
            data_dir=FLAGS.data_dir,
            data_dir_flow=FLAGS.data_dir_flow,
            dataset_name=FLAGS.dataset_name,
            input_height=FLAGS.input_height,
            input_width=FLAGS.input_width,
            output_height=FLAGS.output_height,
            output_width=FLAGS.output_width,
            is_crop=FLAGS.is_crop,
            batch_size=FLAGS.batch_size,
            c_dim=FLAGS.c_dim,
            checkpoint_dir=FLAGS.checkpoint_dir)

    if FLAGS.is_train:
        flowgan.train(FLAGS)
    else:
        if not flowgan.load(FLAGS.checkpoint_dir):
            raise Exception("[!] Train a model first, then run test mode")

    # Below is codes for visualization
    # OPTION = 1
    # visualize(sess, flowgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()
