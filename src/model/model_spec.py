
"""
This platform will include most of regular semantic-segmentation network as well as regular classification network. 
Hope using this platform can do distributed training on GCP MLE.

Done:
Semantic Segmentation network

To Do:
Classification network

"""
from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os
import sys

import numpy as np
import tensorflow as tf

from model import model_builder

from model.model_utils import preprocessing, pretrain_model_manager,get_iou

tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags
FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        features = features['feature']

    images = tf.cast(
        tf.map_fn(preprocessing.mean_image_addition, features),
        tf.uint8)

    logits = model_builder.build_model(model_name=params['model_name'],
                                       frontend=params['base_architecture'],
                                       net_input=images,
                                       num_classes=params['num_classes'],
                                       crop_width=params['crop_width'],
                                       crop_height=params['crop_height'],
                                       is_training=(
                                           mode == tf.estimator.ModeKeys.TRAIN)
                                       )
    
    # if upsample_logits is true, will resize logits to image orignal size
    # otherwise, label will do resize outside to be like logtis size
    if FLAGS.upsample_logits:
        # Label is not downsampled, and instead we upsample logits.
        logits = tf.image.resize_bilinear(logits, tf.shape(images)[1:3], align_corners=True)
    else:
        # Label is downsampled to the same size as logits.
        labels = tf.image.resize_nearest_neighbor(labels, tf.shape(logits)[1:3], align_corners=True)

    pred_classes = tf.expand_dims(
        tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

    pred_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                     [pred_classes, params['batch_size'],
                                         params['num_classes']],
                                     tf.uint8)

    predictions = {
        'classes': pred_classes,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'decoded_labels': pred_decoded_labels
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
        predictions_without_decoded_labels = predictions.copy()
        del predictions_without_decoded_labels['decoded_labels']

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'preds': tf.estimator.export.PredictOutput(
                    predictions_without_decoded_labels)
            })

    gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                   [labels, params['batch_size'], params['num_classes']], tf.uint8)

    labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.
    logtis_flat = tf.reshape(logits, [-1, params['num_classes']])
    labels_flat = tf.reshape(labels, [-1, ])
    preds_flat = tf.reshape(pred_classes, [-1, ])
    
    confusion_matrix = tf.confusion_matrix(labels_flat, preds_flat, num_classes=params['num_classes'])

    predictions['preds_flat'] = preds_flat
    predictions['labels_flat'] = labels_flat
    predictions['confusion_matrix'] = confusion_matrix

    weights = tf.zeros_like(labels_flat,dtype=tf.float32)

    for i in range(params['num_classes']):
        weights+=tf.to_float(tf.equal(labels_flat,i))*params['class_weights'][i]
    
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logtis_flat, labels=labels_flat,weights=weights)


    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    # tf.summary.scalar('cross_entropy1',cross_entropy1)

    if not params['freeze_batch_norm']:
        train_var_list = [v for v in tf.trainable_variables()]
    else:
        train_var_list = [v for v in tf.trainable_variables()
                          if 'beta' not in v.name and 'gamma' not in v.name]

    # Add weight decay to the loss.
    with tf.variable_scope("total_loss"):
        loss = cross_entropy + params.get('weight_decay', FLAGS.weight_decay) * tf.add_n(
            [tf.nn.l2_loss(v) for v in train_var_list])
    # loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.image('images',
                         tf.concat(axis=2, values=[
                                   images, gt_decoded_labels, pred_decoded_labels]),
                         max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

        global_step = tf.train.get_or_create_global_step()

        if params['learning_rate_policy'] == 'piecewise':
            # Scale the learning rate linearly with the batch size. When the batch size
            # is 128, the learning rate should be 0.1.
            initial_learning_rate = 0.1 * params['batch_size'] / 128
            batches_per_epoch = params['num_train'] / params['batch_size']
            # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
            boundaries = [int(batches_per_epoch * epoch)
                          for epoch in [100, 150, 200]]
            values = [initial_learning_rate *
                      decay for decay in [1, 0.1, 0.01, 0.001]]
            learning_rate = tf.train.piecewise_constant(
                tf.cast(global_step, tf.int32), boundaries, values)
        elif params['learning_rate_policy'] == 'poly':
            learning_rate = tf.train.polynomial_decay(
                params['initial_learning_rate'],
                tf.cast(global_step, tf.int32) - params['initial_global_step'],
                params['max_iter'], params['end_learning_rate'], power=params['power'])
        else:
            raise ValueError(
                'Learning rate policy must be "piecewise" or "poly"')

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        # optimizer = tf.train.MomentumOptimizer(
        #     learning_rate=learning_rate,
        #     momentum=params['momentum'])

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)

        grad_vars = optimizer.compute_gradients(loss)
        minimize_op = optimizer.apply_gradients(grad_vars, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        grad_update_ops = model_builder.deeplabV3PlusOfficial_multi_gradient(
            optimizer, loss)
        update_ops.append(grad_update_ops)
        # Batch norm requires update ops to be added as a dependency to the train_op
        train_op = tf.group(minimize_op,update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        labels_flat, preds_flat)

    mean_iou = tf.metrics.mean_iou(
        labels_flat, preds_flat, params['num_classes'])


    # Should do this convertion, when use mirror strategy
    mean_iou = (mean_iou[0], tf.convert_to_tensor(mean_iou[1]))

    metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_px_accuracy')
    tf.summary.scalar('train_px_accuracy', accuracy[1])

    train_mean_iou = get_iou.compute_mean_iou(mean_iou[1])

    tf.identity(train_mean_iou, name='train_mean_iou')
    tf.summary.scalar('train_mean_iou', train_mean_iou)

    for model_var in tf.trainable_variables():
        tf.summary.histogram(model_var.op.name, model_var)

    var,ckpt=pretrain_model_manager.get_model_init_fn(FLAGS.Log_dir, FLAGS.model_name, FLAGS.base_architecture, FLAGS.fine_tune, FLAGS.use_pretrained_model)
    pretrain_model_manager.init_from_checkpoint(var,ckpt)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )
