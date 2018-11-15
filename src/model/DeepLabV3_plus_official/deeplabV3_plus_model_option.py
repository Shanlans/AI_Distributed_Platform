import collections




modelOptions = collections.namedtuple('ModelOptions', ['outputs_to_num_classes',
                                                       'crop_size',
                                                       'atrous_rates',
                                                       'output_stride',
                                                       'merge_method',
                                                       'add_image_level_feature',
                                                       'aspp_with_batch_norm',
                                                       'aspp_with_separable_conv',
                                                       'multi_grid',
                                                       'decoder_output_stride',
                                                       'decoder_use_separable_conv',
                                                       'logits_kernel_size',
                                                       'model_variant',
                                                       'depth_multiplier', 
                                                       'image_pyramid',
                                                       'last_layer_gradient_multiplier',
                                                       'last_layers_contain_logits_only',
                                                       'initialize_last_layer',
                                                       'upsample_logits',
                                                       'class_weights',
                                                       ])


# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'

xception65_option = modelOptions(
    outputs_to_num_classes={OUTPUT_TYPE: 5},

    # 'Image crop size [height, width] during training.'
    crop_size=[512, 512],

    # 'Atrous rates for atrous spatial pyramid pooling.'
    # For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
    # rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
    # one could use different atrous_rates/output_stride during training/evaluation.
    atrous_rates=[6, 12, 18],

    # 'The ratio of input to output spatial resolution.'
    output_stride=16,

    # ['max', 'avg'], 'Scheme to merge multi scale features.'
    merge_method='max',

    # 'Add image level feature.'Global Average Pooling Layer and resize layer need crop image size information.
    add_image_level_feature=True,

    # 'Use batch norm parameters for ASPP or not.'
    aspp_with_batch_norm=True,

    # 'Use separable convolution for ASPP or not.'
    aspp_with_separable_conv=True,

    # 'Employ a hierarchy of atrous rates for ResNet.'
    multi_grid=None,

    # 'The ratio of input to output spatial resolution when employing decoder to refine segmentation results.'
    # For `xception_65`, use decoder_output_stride = 4. For `mobilenet_v2`, use
    # decoder_output_stride = None.
    decoder_output_stride=4,

    #'Employ separable convolution for decoder or not.'
    decoder_use_separable_conv=True,

    #'The kernel size for the convolutional kernel that generates logits.'
    logits_kernel_size=1,

    # ['xception_65', 'xception_41','mobilenet_v2'], 'DeepLab model variant.'
    model_variant='xception_65',

    # 'Multiplier for the depth (number of channels) for all convolution ops used in MobileNet.'
    depth_multiplier=1.0,

    # Input image scales for multi-scale feature extraction.
    image_pyramid = None,

    # The gradient multiplier for last layers, which is used to boost the gradient of last layers if the value > 1.
    last_layer_gradient_multiplier = 10.0,

    # Only consider logits as last layers or not
    # Currently, if False will do gradient multiply on _LOGITS_SCOPE_NAME,_IMAGE_POOLING_SCOPE,_ASPP_SCOPE,_CONCAT_PROJECTION_SCOPE,_DECODER_SCOPE,
    # Otherwise, will do only on last layer _LOGITS_SCOPE_NAME
    last_layers_contain_logits_only = False,

    # Set to False if one does not want to re-use the trained classifier weights.
    initialize_last_layer = False,

    # Class_weights, length should be equal to class number 
    class_weights = [1,5,5,5,5],

    # Upsample logits during training
    upsample_logits = True
)
