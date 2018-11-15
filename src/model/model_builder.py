import sys
import os
import tensorflow as tf
import subprocess

from model_utils.pretrain_model_manager import get_model_init_fn,prepare_pretrain_model

from segmodels.FC_DenseNet_Tiramisu import build_fc_densenet
from segmodels.Encoder_Decoder import build_encoder_decoder
from segmodels.RefineNet import build_refinenet
from segmodels.FRRN import build_frrn
from segmodels.MobileUNet import build_mobile_unet
from segmodels.PSPNet import build_pspnet
from segmodels.GCN import build_gcn
from segmodels.DeepLabV3 import build_deeplabv3
from segmodels.DeepLabV3_plus import build_deeplabv3_plus
from segmodels.AdapNet import build_adaptnet
from segmodels.custom_model import build_custom
from segmodels.DenseASPP import build_dense_aspp

from DeepLabV3_plus_official.build_deepLabV3_plus_official import build_deeplabv3_plus as deeplabv3_plus_official
from DeepLabV3_plus_official.build_deepLabV3_plus_official import get_extra_layer_scopes
from DeepLabV3_plus_official.deeplabV3_plus_model_option import xception65_option
from DeepLabV3_plus_official.train_utils import get_model_gradient_multipliers



SUPPORTED_MODELS = ["FC-DenseNet56", "FC-DenseNet67", "FC-DenseNet103", "Encoder-Decoder", "Encoder-Decoder-Skip", "RefineNet",
                    "FRRN-A", "FRRN-B", "MobileUNet", "MobileUNet-Skip", "PSPNet", "GCN", "DeepLabV3", "DeepLabV3_plus", "DeepLabV3_plus_official", "AdapNet",
                    "DenseASPP", "custom"]

SUPPORTED_FRONTENDS = ["ResNet50", "ResNet101",
                       "ResNet152", "MobileNetV2", "InceptionV4", "None"]






def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="ResNet101", is_training=True):
    # Get the selected model.
    # Some of them require pre-trained ResNet

    print("Preparing the model ...")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            "The model you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

    if frontend not in SUPPORTED_FRONTENDS:
        raise ValueError("The frontend you selected is not supported. The following models are currently supported: {0}".format(
            SUPPORTED_FRONTENDS))

    prepare_pretrain_model(model_name,frontend)

    network = None
    if model_name == "FC-DenseNet56" or model_name == "FC-DenseNet67" or model_name == "FC-DenseNet103":
        network = build_fc_densenet(
            net_input, preset_model=model_name, num_classes=num_classes)
    elif model_name == "RefineNet":
        # RefineNet requires pre-trained ResNet weights
        network = build_refinenet(
            net_input, preset_model=model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
    elif model_name == "FRRN-A" or model_name == "FRRN-B":
        network = build_frrn(
            net_input, preset_model=model_name, num_classes=num_classes)
    elif model_name == "Encoder-Decoder" or model_name == "Encoder-Decoder-Skip":
        network = build_encoder_decoder(
            net_input, preset_model=model_name, num_classes=num_classes)
    elif model_name == "MobileUNet" or model_name == "MobileUNet-Skip":
        network = build_mobile_unet(
            net_input, preset_model=model_name, num_classes=num_classes)
    elif model_name == "PSPNet":
        # Image size is required for PSPNet
        # PSPNet requires pre-trained ResNet weights
        network = build_pspnet(net_input, label_size=[
            crop_height, crop_width], preset_model=model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
    elif model_name == "GCN":
        # GCN requires pre-trained ResNet weights
        network = build_gcn(net_input, preset_model=model_name,
                            frontend=frontend, num_classes=num_classes, is_training=is_training)
    elif model_name == "DeepLabV3":
        # DeepLabV requires pre-trained ResNet weights
        network = build_deeplabv3(
            net_input, preset_model=model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
    elif model_name == "DeepLabV3_plus":
        # DeepLabV3+ requires pre-trained ResNet weights
        network = build_deeplabv3_plus(
            net_input, preset_model=model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
    elif model_name == "DeepLabV3_plus_official":
        network = deeplabv3_plus_official(
            net_input, xception65_option, weight_decay=0.0001, is_training=False, fine_tune_batch_norm=True)
    elif model_name == "DenseASPP":
        # DenseASPP+ requires pre-trained ResNet weights
        network = build_dense_aspp(
            net_input, preset_model=model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
    elif model_name == "AdapNet":
        network = build_adaptnet(net_input, num_classes=num_classes)
    elif model_name == "custom":
        network = build_custom(net_input, num_classes)
    else:
        raise ValueError(
            "Error: the model %d is not available. Try checking which models are available using the command python main.py --help")
    return network

def deeplabV3PlusOfficial_multi_gradient(optimizer,loss):
    # Optimizer: Like tf.train.Adagrad....
    # Loss: the loss you know...
    # return: The updated list of gradient to variable pairs.
    # Like this [(<tf.Tensor 'gradients/[operation name]:[How large will be multiplied]'...>,<tf.Variable ...>),(<...>,<...>),(<...>,<...>)...]

    # To Do:
    # In the future, will can be the regular function for other networks

    # Modify the gradients for biases and last layer variables.
    gra_and_var = optimizer.compute_gradients(loss)    
    last_layers = get_extra_layer_scopes(xception65_option.last_layers_contain_logits_only)
    grad_mult = get_model_gradient_multipliers(last_layers, xception65_option.last_layer_gradient_multiplier)

    grad_update_ops = tf.contrib.training.multiply_gradients(gra_and_var,grad_mult)

    return grad_update_ops