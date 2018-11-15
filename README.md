# AI_Training_Platform

*This work is inspired by [GeorgeSeif's](https://github.com/GeorgeSeif) work.  
[Tensorflow](https://www.tensorflow.org/) doesn't like pytorch can do distributed training easily, we has to use the [estimator](https://www.tensorflow.org/guide/estimators) to get the distributed training target. Meanwhile, popular semantic networks usually are based on the classification feature extract model, so I want to implement classification and segmentation together that user can not only choose the trainning task(Cls or Seg), but also choose the feature extraction model with classifier or Segmentation network. The most important thing is we can run all of these on distributed system*

### Model
**Model includes two levels, one is front-end (or feature extraction). Another is segmentation network which is based on front-end**
- [Front-end](./src/model/frontends) Dedicated to implement all of popular feature extraction model.
  
  ##### *Finish Testing*
    
  - [ ] [Resnet_V1](./src/model/frontends/resnet_v1.py) 
  - [X] [Resnet_V2](./src/model/frontends/resnet_v2.py)
  - [X] [Xception](./src/model/frontends/xception.py)
  - [ ] [MobileNet-V1](./src/model/frontends/mobilenet_v1.py)
  - [X] [MobileNet-V2](./src/model/frontends/mobilenet_v2.py)
  - [ ] [Incetion_V1](./src/model/frontends/inception_v1.py)
  - [ ] [Incetion_V2](./src/model/frontends/inception_v2.py)
  - [ ] [Incetion_V3](./src/model/frontends/inception_v3.py)
  - [X] [Incetion_V4](./src/model/frontends/inception_v4.py)
  - [ ] [Se_Resnent](./src/model/frontends/se_resnext.py)
  - [X] [Vgg](./src/model/frontends/vgg.py)
  - [ ] [DCGAN](./src/model/frontends/dcgan.py)  
  - [ ] [AlexNet](./src/model/frontends/alexnet.py)
  - [ ] [cifarNet](./src/model/frontends/cifarnet.py)
  
- [Segmentation](./src/model/segmodels) Dedicated to implement all of popular Semantic Segmentation model.
  
  ##### *Finish Testing*
  
  - [ ] [AdapNet](./src/model/segmodels/AdapNet.py)
  - [ ] [DeepLabV3](./src/model/segmodels/DeepLabV3.py)
  - [X] [DeepLabV3Plus](./src/model/segmodels/DeepLabV3_plus.py)
  - [X] [DenseASPP](./src/model/segmodels/DenseASPP.py)
  - [X] [Tiramisu](./src/model/segmodels/FC_DenseNet_Tiramisu.py)
  - [X] [MobileUnet](./src/model/segmodels/MobileUNet.py)
  - [X] [Encoder-Decoder](./src/model/segmodels/Encoder_Decoder.py)
  
  
 

 

