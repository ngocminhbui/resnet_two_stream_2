from resnet_train import train
from resnet_architecture import *
from exp_config import *
import tensorflow as tf
import os
from input import distorted_inputs

FLAGS = tf.app.flags.FLAGS


def main(_):
    images_rgb,image_depth, labels = distorted_inputs(FLAGS.data_dir, FLAGS.train_lst)

    is_training = tf.placeholder('bool', [], name='is_training')  # placeholder for the fusion part

    logits = inference(images_rgb,image_depth,
                       num_classes=FLAGS.num_classes,
                       is_training=is_training,
                       num_blocks=[3, 4, 6, 3])
    train(is_training,logits, images_rgb,image_depth , labels)


if __name__ == '__main__':
    tf.app.run(main)
