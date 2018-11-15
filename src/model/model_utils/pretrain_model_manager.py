import subprocess
import tarfile
import urllib
import os
import shutil

import tensorflow as tf

slim = tf.contrib.slim

from model.DeepLabV3_plus_official import deeplabV3_plus_model_option,build_deepLabV3_plus_official


_DOWNLOAD_LINK = {
    'ResNet50': 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
    'ResNet101': 'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
    'ResNet152': 'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
    'MobileNetV2': 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz',
    'InceptionV4': 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
    'NASNet': 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz',
    'DeepLabV3PlusOfficial': 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz'
}

_MODEL_PATH = {
    'ResNet50': 'pretrain_models/ResNet50',
    'ResNet101': 'pretrain_models/ResNet101',
    'ResNet152': 'pretrain_models/ResNet152',
    'MobileNetV2': 'pretrain_models/MobileNetV2',
    'InceptionV4': 'pretrain_models/InceptionV4',
    'NASNet': 'pretrain_models/NASNet',
    'DeepLabV3PlusOfficial': 'pretrain_models/DeepLabV3PlusOfficial'
}

_CHECK_POINT_PATH = {
    'ResNet50': "pretrain_models/ResNet50/resnet_v2_50.ckpt",
    'ResNet101': "pretrain_models/ResNet50/resnet_v2_101.ckpt",
    'ResNet152': "pretrain_models/ResNet152/resnet_v2_152.ckpt",
    'MobileNetV2': "pretrain_models/MobileNetV2/mobilenet_v2_1.4_224.ckpt.data-00000-of-00001",
    'InceptionV4': "pretrain_models/InceptionV4/inception_v4.ckpt",
    'NASNet': "pretrain_models/NASNet/model.ckpt.data-00000-of-00001",
    'DeepLabV3_plus_official': "pretrain_models/DeepLabV3PlusOfficial/deeplabv3_pascal_train_aug/model.ckpt"
}

def download_checkpoints(model):
    path = "pretrain_models"
    url = _DOWNLOAD_LINK[model]
    model_path = _MODEL_PATH[model]
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    try:
        tars, _ = urllib.request.urlretrieve(url)
        thetarfile = tarfile.open(name=tars, mode="r|gz")
        thetarfile.extractall(model_path)
        thetarfile.close()
    except Exception as e:
        print(e)
        pass


def prepare_pretrain_model(model_name, frontend):
    try:
        if "ResNet50" == frontend and not os.path.isfile(_CHECK_POINT_PATH["ResNet50"]):
            download_checkpoints("ResNet50")
        if "ResNet101" == frontend and not os.path.isfile(_CHECK_POINT_PATH["ResNet101"]):
            download_checkpoints("ResNet101")
        if "ResNet152" == frontend and not os.path.isfile(_CHECK_POINT_PATH["ResNet152"]):
            download_checkpoints("ResNet152")
        if "MobileNetV2" == frontend and not os.path.isfile(_CHECK_POINT_PATH["MobileNetV2"]):
            download_checkpoints("MobileNetV2")
        if "InceptionV4" == frontend and not os.path.isfile(_CHECK_POINT_PATH["InceptionV4"]):
            download_checkpoints("InceptionV4")
        if "NASNet" == frontend and not os.path.isfile(_CHECK_POINT_PATH["NASNet"]):
            download_checkpoints("NASNet")
        if "DeepLabV3_plus_official" == model_name and not os.path.isfile(_CHECK_POINT_PATH["DeepLabV3_plus_official"]+'.data-00000-of-00001'):
            download_checkpoints("DeepLabV3PlusOfficial")
    except Exception as e:
        print(e)
        pass


def get_model_init_fn(log_dir, model, frontend,fine_tune,use_pretrained_model):
    """Gets the function initializing model variables from a checkpoint.
    Args:
    log_dir: Log directory for training.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    Returns:
    Initialization function.
    """
    # Variables that will not be restored.

    if model == 'DeepLabV3_plus_official':
	    last_layers = build_deepLabV3_plus_official.get_extra_layer_scopes(deeplabV3_plus_model_option.xception65_option.last_layers_contain_logits_only)
	    exclude_list = ['global_step']
	    if not deeplabV3_plus_model_option.xception65_option.initialize_last_layer: 
	        exclude_list.extend(last_layers)
	    restore_variables = slim.get_variables_to_restore(exclude=exclude_list)
    else:
        if frontend == 'ResNet50':
            restore_variables = slim.get_model_variables('resnet_v2_50')
        elif frontend == 'ResNet101':
            restore_variables = slim.get_model_variables('resnet_v2_101')
        elif frontend == 'ResNet152':
            restore_variables = slim.get_model_variables('resnet_v2_152')
        elif frontend == 'MobileNetV2':
            restore_variables = slim.get_model_variables('mobilenet_v2')
        elif frontend == 'InceptionV4':
            restore_variables = slim.get_model_variables('inception_v4')
        else:
            raise ValueError(
                "Unsupported fronetnd model '%s'. This function only supports ResNet50, ResNet101, ResNet152, and MobileNetV2" % (frontend))
	
    if fine_tune:
	    if tf.train.get_checkpoint_state(log_dir):
	        tf.logging.info('Ignoring initialization; other checkpoint exists')
	        ckpt_path = tf.train.latest_checkpoint(log_dir)
	    else:
	        tf.logging.info('Checkpoint not exists, cannot fine tune')
	        raise ValueError('Please check the checkpoint folder')

	    restore_variables = slim.get_variables_to_restore()
	    

    elif use_pretrained_model:
	    tf.logging.info('Initializing model from path: %s',_CHECK_POINT_PATH[model])
	    ckpt_path = _CHECK_POINT_PATH[model]
    else:
	    ckpt_path = None
	    restore_variables = None
    
    return restore_variables,ckpt_path
    
def init_from_checkpoint(restore_variables,ckpt_path):
    if restore_variables:
        variable_list = {v.name.split(':')[0]: v for v in restore_variables if 'Adam' not in v.name and 'beta1' not in v.name and 'beta2' not in v.name}#if 'beta' not in v.name and 'gamma' not in v.name}
    
    tf.train.init_from_checkpoint(ckpt_path,variable_list)
