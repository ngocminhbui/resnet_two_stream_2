from resnet_train import train
from resnet_architecture import *
import tensorflow as tf
import os
import numpy as np

FLAGS = tf.app.flags.FLAGS

''' Load list of  {filename, label_name, label_index} '''
def load_data(data_dir, data_lst):
    data = []
    train_lst = open(data_lst, 'r').read().splitlines()
    dictionary = open(FLAGS.dictionary, 'r').read().splitlines()
    for img_fn in train_lst:
        fn = os.path.join(data_dir, img_fn + '_crop.png')
        fn_depth = os.path.join(data_dir, img_fn + '_depthcrop.png')
        label_name = img_fn.split('/')[0]
        label_index = dictionary.index(label_name)
        data.append({
            "filename": fn,
            "filename_depth": fn_depth,
            "label_name": label_name,
            "label_index": label_index
        })
    return data


''' Load input data using queue (feeding)'''


mean_im_rgb = np.load(FLAGS.mean_rgb)
mean_im_rgb = mean_im_rgb[..., ::-1]

mean_im_depth = np.load(FLAGS.mean_dep)
mean_im_depth = mean_im_depth[..., ::-1]


def read_image_from_disk(input_queue):
    global mean_im_rgb
    global mean_im_depth

    label = input_queue[2]
    file_contents_rgb = tf.read_file(input_queue[0])
    file_contents_depth = tf.read_file(input_queue[1])

    example_rgb = tf.image.decode_png(file_contents_rgb, channels=3)
    example_depth = tf.image.decode_png(file_contents_depth, channels=3)

    example_rgb = tf.cast(example_rgb, tf.float32) - tf.convert_to_tensor(mean_im_rgb, dtype=tf.float32)
    example_depth = tf.cast(example_depth, tf.float32) - tf.convert_to_tensor(mean_im_depth, dtype=tf.float32)

    return example_rgb,example_depth, label


def distorted_inputs(data_dir, data_lst):
    data = load_data(data_dir, data_lst)

    filenames = [ d['filename'] for d in data ]
    filenames_depth = [ d['filename_depth'] for d in data ]

    label_indexes = [ d['label_index'] for d in data ]

    input_queue = tf.train.slice_input_producer([filenames,filenames_depth, label_indexes], shuffle=True)

    # read image and label from disk
    image_rgb, image_depth, label = read_image_from_disk(input_queue)


    ''' Data Augmentation '''
    image_rgb = tf.random_crop(image_rgb, [FLAGS.input_size, FLAGS.input_size, 3])
    image_rgb = tf.image.random_flip_left_right(image_rgb)

    image_rgb = tf.image.random_brightness(image_rgb, max_delta=32. / 255.)
    image_rgb = tf.image.random_saturation(image_rgb, lower=0.5, upper=1.5)
    image_rgb = tf.image.random_hue(image_rgb, max_delta=0.2)
    image_rgb = tf.image.random_contrast(image_rgb, lower=0.5, upper=1.5)
    image_rgb = tf.image.per_image_standardization(image_rgb)

    image_depth = tf.random_crop(image_depth, [FLAGS.input_size, FLAGS.input_size, 3])
    image_depth = tf.image.random_flip_left_right(image_depth)
    image_depth = tf.image.random_brightness(image_depth, max_delta=32. / 255.)
    image_depth = tf.image.random_saturation(image_depth, lower=0.5, upper=1.5)
    image_depth = tf.image.random_hue(image_depth, max_delta=0.2)
    image_depth = tf.image.random_contrast(image_depth, lower=0.5, upper=1.5)
    image_depth = tf.image.per_image_standardization(image_depth)

    # generate batch

    image_rgb_batch,image_depth_batch, label_batch = tf.train.shuffle_batch(
        [image_rgb, image_depth, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocess_threads,
        capacity=FLAGS.min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.min_queue_examples)

    return image_rgb_batch,image_depth_batch, tf.reshape(label_batch, [FLAGS.batch_size])


def inputs(data_dir, data_lst):
    data = load_data(data_dir, data_lst)

    filenames = [ d['filename'] for d in data ]
    filenames_depth = [ d['filename_depth'] for d in data ]

    label_indexes = [ d['label_index'] for d in data ]

    input_queue = tf.train.slice_input_producer([filenames,filenames_depth, label_indexes], shuffle=False)

    # read image and label from disk
    image_rgb, image_depth, label = read_image_from_disk(input_queue)


    ''' Data Augmentation '''
    image_rgb = tf.random_crop(image_rgb, [FLAGS.input_size, FLAGS.input_size, 3])
    image_rgb = tf.image.random_flip_left_right(image_rgb)

    image_rgb = tf.image.random_brightness(image_rgb, max_delta=32. / 255.)
    image_rgb = tf.image.random_saturation(image_rgb, lower=0.5, upper=1.5)
    image_rgb = tf.image.random_hue(image_rgb, max_delta=0.2)
    image_rgb = tf.image.random_contrast(image_rgb, lower=0.5, upper=1.5)
    image_rgb = tf.image.per_image_standardization(image_rgb)

    image_depth = tf.random_crop(image_depth, [FLAGS.input_size, FLAGS.input_size, 3])
    image_depth = tf.image.random_flip_left_right(image_depth)
    image_depth = tf.image.random_brightness(image_depth, max_delta=32. / 255.)
    image_depth = tf.image.random_saturation(image_depth, lower=0.5, upper=1.5)
    image_depth = tf.image.random_hue(image_depth, max_delta=0.2)
    image_depth = tf.image.random_contrast(image_depth, lower=0.5, upper=1.5)
    image_depth = tf.image.per_image_standardization(image_depth)

    # generate batch

    image_rgb_batch,image_depth_batch, label_batch = tf.train.shuffle_batch(
        [image_rgb, image_depth, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocess_threads,
        capacity=FLAGS.min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.min_queue_examples)

    return image_rgb_batch,image_depth_batch, tf.reshape(label_batch, [FLAGS.batch_size])

