import numpy as np
import os
import time
import sys

from resnet_architecture import *
from exp_config import *
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

def train(is_training, logits, images_rgb,images_depth, labels):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)


    loss_ = loss(logits, labels)
    predictions = tf.nn.softmax(logits)
    top1_error = top_k_error(predictions, labels, 1)


    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    learning_rate = tf.train.exponential_decay(FLAGS.starter_learning_rate,
                                               global_step,
                                               decay_steps=FLAGS.train_decay_steps,
                                               decay_rate=FLAGS.train_decay_rate,
                                               staircase=True)


    tf.summary.scalar('learning_rate', learning_rate)

    opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.image_summary('images', images_rgb)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print 'No checkpoint to continue from in', FLAGS.train_dir
            sys.exit(1)
        print 'resume', latest
        saver.restore(sess, latest)
    else:
        print 'Restore from pretrained model..',FLAGS.pretrained_rgb_model,FLAGS.pretrained_depth_model
        print "Reloading weights for encoders..."
        net_rgb = np.load(FLAGS.pretrained_rgb_model).item()
        net_depth = np.load(FLAGS.pretrained_depth_model).item()
        for v in tf.trainable_variables():
            t_name = v.name  # get name of tensor variable

            # get value of corresponding net
            if t_name in net_rgb.keys():
                n_value = net_rgb[t_name]
                print t_name, 'from rgb'
            elif t_name in net_depth.keys():
                n_value = net_depth[t_name]
                print t_name, 'from depth'
            else:
                print 'Do not exist key ', t_name
            # avoid replacing variables with different shape (e.g. last fc layer)
            if tuple(v.get_shape().as_list()) == n_value.shape:
                sess.run(v.assign(n_value))
            else:
                print '\t%s is not reinitialized' % t_name

    print 'start training..'
    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)

        o = sess.run(i, { is_training: True })

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 3000 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        if step > 1 and step % 100 == 0:
            _, top1_error_value = sess.run([val_op, top1_error], { is_training: False })
            print('Validation top1 error %.2f' % top1_error_value)
