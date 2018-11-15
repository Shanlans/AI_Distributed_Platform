# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Create Seg AI platform using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import glob
import pdb
import subprocess


sys.path.append("/root/.local/lib/python3.5/site-packages/sunny_ai_platform")
sys.path.append("/root/.local/lib/python3.5/site-packages/sunny_ai_platform/dataset")
sys.path.append("/root/.local/lib/python3.5/site-packages/sunny_ai_platform/model")
sys.path.append("/root/.local/lib/python3.5/site-packages/sunny_ai_platform/utils")

# sys.path.append('dataset')
# sys.path.append('model')
# sys.path.append('utils')

import tensorflow as tf
from sklearn.model_selection import train_test_split



from model.model_utils import preprocessing
from dataset.data_merge import download
from utils.misc import distribution_utils
from utils.logs import hooks_helper
from model import model_spec

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('Log_dir',
                    './Log',
                    'Directory to save model parameters, graph and etc.'
                    'This can also be used to load checkpoints from the directory'
                    'into an estimator to continue training a previously saved model.')


# flags.DEFINE_boolean('clean_model_dir',
#                      True,
#                      'Whether to clean up the model directory if present.'
#                      'If True, will delete model file, do training startover')

flags.DEFINE_enum('model_name',
                  'DeepLabV3_plus_official',
                  ["FC-DenseNet56", "FC-DenseNet67", "FC-DenseNet103", "Encoder-Decoder", "Encoder-Decoder-Skip", "RefineNet",
                   "FRRN-A", "FRRN-B", "MobileUNet", "MobileUNet-Skip", "PSPNet", "GCN", "DeepLabV3", "DeepLabV3_plus",
                   "DeepLabV3_plus_official", "AdapNet", "DenseASPP", "custom"],
                  'Now, only "DeepLabV3_plus_official" is ready to test')

flags.DEFINE_enum('base_architecture',
                  "None",
                  ["ResNet50", "ResNet101", "ResNet152",
                      "MobileNetV2", "InceptionV4", "None"],
                  'This arg to choose the feature extractor for semantic segmentation except "DeepLabV3_plus_official", '
                  ' "DeepLabV3_plus_official" choose Xception65 automatically.')


flags.DEFINE_multi_integer('crop_size',
                           [512, 512],
                           'Image crop size [height, width] during training')

flags.DEFINE_integer('batch_size',
                     2,
                     'Number of examples per batch')

flags.DEFINE_integer('train_epochs',
                     5,
                     'Number of training epochs')

flags.DEFINE_integer('epochs_per_eval',
                     1,
                     'The number of training epochs to run between evaluations.')

flags.DEFINE_enum('learning_rate_policy',
                  'poly',
                  ['poly', 'piecewise'],
                  'Learning rate policy to optimize loss.')

flags.DEFINE_integer('max_iter',
                     300000,
                     'Number of maximum iteration used for "poly" learning rate policy.')


flags.DEFINE_integer('tensorboard_images_max_outputs',
                     1,
                     'Max number of batch elements to generate for Tensorboard.')


# flags.DEFINE_string('data_dir',
#                     'D:\\pythonworkspace\\SSL\\models\\research\\deeplab\\datasets\\IRholder6\\tfrecord',
#                     'Path to the directory containing the data tf record.')


flags.DEFINE_bool('use_pretrained_model',
                  True,
                  'Use pretrain official model')

flags.DEFINE_bool('fine_tune',
                  False,
                  'Use trained checkpoint to do fine tune')

flags.DEFINE_boolean('freeze_batch_norm',
                     True,
                     'Freeze batch normalization parameters during the training.')

flags.DEFINE_float('initial_learning_rate',
                   1e-5,
                   'Initial learning rate for the optimizer.')

flags.DEFINE_float('end_learning_rate',
                   1e-12,
                   'End learning rate for the optimizer.')

flags.DEFINE_integer('initial_global_step',
                     0,
                     'Initial global step for controlling learning rate when fine-tuning model.')

flags.DEFINE_float('weight_decay',
                   5e-4,
                   'The weight decay to use for regularizing the model.')

flags.DEFINE_boolean('debug',
                     False,
                     'Whether to use debugger to track down bad values during training.')

flags.DEFINE_multi_integer('class_weights',
                           [1,5,5,5,5],
                           'weighted class')

flags.DEFINE_boolean('upsample_logits',
                    True,
                    'If True, will upsample the logits, otherwise, will downsample the label to the same size as logits')


flags.DEFINE_float('decay_power',
                    0.9,
                    'Learning rate decay power')


###For customized###
flags.DEFINE_string('job-dir', None, 'mle job dir')
flags.DEFINE_string('input_path', None, 'dataset input')
flags.DEFINE_string('output_path', None, 'output path')

flags.DEFINE_string('param_epoch', None, 'Epochs number')
flags.DEFINE_string('param_batch_size',None,'Batch size')

flags.DEFINE_string('param_validate_set_ratio', None, 'validate set number')
flags.DEFINE_string('augmentation_rotation_range',None, 'augmentation rotation range')
flags.DEFINE_string('augmentation_horizontal_flip',None, 'augmentation horizontal flip')
flags.DEFINE_string('augmentation_vertical_flip',None, 'augmentation vertical flip')
flags.DEFINE_string('augmentation_random_brightness',None, 'augmentation random brightness')


_GPU_NUM = 4


def input_fn(is_training, data_tuple,buffer_size, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_tuple: The tuple, ([image_path_list],[image_label_list])
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      A tuple of images and labels.
    """

    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)


    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(
        lambda filename, label: preprocessing.preprocess_image(filename, label, is_training))
    dataset = dataset.prefetch(batch_size)

    # # We call repeat after shuffling, rather than before, to prevent separate
    # # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset

def platform_main(train_data,val_data,class_num):
    # Parameter Get
    train_num = len(train_data[0])
    val_num = len(val_data[0])

    batch_size = int(FLAGS.param_batch_size)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    session_config = tf.ConfigProto(
        allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # Windows not support nccl, refer to https://github.com/tensorflow/tensorflow/issues/21470
    # Only can use hierachical_copy
    distribution_strategy = distribution_utils.get_distribution_strategy(
        _GPU_NUM, 'hierachical_copy')

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy, session_config=session_config)
    # run_config = tf.estimator.RunConfig(session_config=session_config).replace(save_checkpoints_secs=1e9)
    model = tf.estimator.Estimator(
        model_fn=model_spec.model_fn,
        model_dir=FLAGS.Log_dir,
        config=run_config,
        params={
            'model_name': FLAGS.model_name,
            'Log_dir': FLAGS.Log_dir,
            'base_architecture': FLAGS.base_architecture,
            'num_classes': class_num,
            'crop_width': FLAGS.crop_size[0],
            'crop_height': FLAGS.crop_size[1],
            'batch_size': distribution_utils.per_device_batch_size(batch_size,_GPU_NUM),
            'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
            'weight_decay': FLAGS.weight_decay,
            'learning_rate_policy': FLAGS.learning_rate_policy,
            'num_train': train_num,
            'initial_learning_rate': FLAGS.initial_learning_rate,
            'max_iter': FLAGS.max_iter,
            'end_learning_rate': FLAGS.end_learning_rate,
            'power': FLAGS.decay_power,
            'freeze_batch_norm': FLAGS.freeze_batch_norm,
            'initial_global_step': FLAGS.initial_global_step,
            'class_weights':FLAGS.class_weights
        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            # 'global_step': 'global_step',
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_px_accuracy': 'train_px_accuracy',
            'train_mean_iou': 'train_mean_iou',
        }


        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)
        train_hooks = [logging_hook]


        eval_hooks = None

        # if FLAGS.debug:
        #     debug_hook = tf_debug.LocalCLIDebugHook()
        #     train_hooks.append(debug_hook)
        #     eval_hooks = [debug_hook]

        tf.logging.info("Start training.")
        
        model.train(
            input_fn=lambda: input_fn(
                True, train_data, distribution_utils.per_device_batch_size(batch_size,_GPU_NUM),train_num, FLAGS.epochs_per_eval),
            hooks=train_hooks,
            # steps=1  # For debug
        )

        tf.logging.info("Start evaluation.")
        # Evaluate the model and print results
        eval_results = model.evaluate(
            # Batch size must be 1 for testing because the images' size differs
            input_fn=lambda: input_fn(False, val_data,batch_size=1,buffer_size=val_num),
            hooks=eval_hooks,
            # steps=1  # For debug
        )
        tf.logging.info(eval_results)




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    tf.logging.info('========Loading Parameters========')
    flags.mark_flag_as_required('input_path')
    flags.mark_flag_as_required('output_path')
    flags.mark_flag_as_required('param_epoch')
    flags.mark_flag_as_required('param_batch_size')
    flags.mark_flag_as_required('param_validate_set_ratio')
    flags.mark_flag_as_required('augmentation_rotation_range')
    flags.mark_flag_as_required('augmentation_horizontal_flip')
    flags.mark_flag_as_required('augmentation_vertical_flip')
    flags.mark_flag_as_required('augmentation_random_brightness')


    # if FLAGS.clean_model_dir:
    #     shutil.rmtree(FLAGS.Log_dir, ignore_errors=True)
    
    input_path = FLAGS.input_path
    output_path = FLAGS.output_path

    try:
        tf.logging.info('======== Get Dataset =========')
        data_list,class_num = download.get_data(input_path, 'train_data')

        train_image_list,val_image_list,train_label_list,val_label_list = train_test_split(*data_list,test_size=0.25, random_state=0)

        train_data_list = (train_image_list,train_label_list)
        val_data_list = (val_image_list,val_label_list)

        platform_main(train_data_list,val_data_list,class_num)

    except Exception as e:
        print(e)

