[![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/shunk031/chainer-skin-lesion-detector/blob/master/LICENSE)
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)

# Automatic Foot Ulcer segmentation Using an Ensemble of Convolutional Neural Networks
Contains the prediction codes for our submission to the Foot Ulcer Segmentation Challenge


## Method
![Project Image](https://github.com/masih4/Foot_Ulcer_Segmentation/blob/main/git_image/method.png)


## Directory structure
```
.
├── code
│   ├── test.py (main run file)
│   ├── gpu_setting.py
│   ├── metric.py
│   └── params_test.py (all configs are here)
│
├── saved_models
│   ├── linknet(LinkNet models should be downloaded from Google drive and placed in this folder)
│   └── unet (U-Net models should be downloaded from Google drive and placed in this folder)
│
├── test images (place all test images here (size 512x512 pixels))
│
└── results
    ├── temp
    └── final (final results will be saved here)
 
```

1- run the following commands inside the container:
```
$ cd src/code
$ python3 test.py 
```
2- final results will be saved inside `results/final` folder

## Results
To derive the results in the following table, we used the Medetec foot ulcer dataset [1] for pre-training. Then we used the training set of the MICCAI 2021 Foot Ulcer Segmentation Challenge dataset [2] (810 images) as the training set. The reported results in the following table are based on the validation set of the Foot Ulcer Segmentation dataset (200 images). For the challenge submssion, we used the entire 1010 images of the train and validation set to train our models. 

| Model                                 | Image-based Dice (%) | Precision (%)    | Recall (%)       | Dataset-based IOU (%)   | Dataset-based Dice (%)     |
| --------------------------------      |:--------------------:|:----------------:|:----------------:|:-----------------------:|:--------------------------:|
| VGG16  [2]                            |         -            |   83.91          |   78.35          |    -                    | 81.03                      |
| SegNet [2]                            |         -            |   83.66          |   86.49          |    -                    | 85.05                      |
| U-Net [2]                             |         -            |   89.04          |   91.29          |    -                    | 90.15                      |
| Mask-RCNN  [2]                        |         -            |   **94.30**      |   86.40          |    -                    | 90.20                      |
| MobileNetV2 [2]                       |         -            |   90.86          |   89.76          |    -                    | 90.30                      |
| MobileNetV2 + pp [2]                  |         -            |   91.01          |   89.97          |    -                    | 90.47                      |
| **EfficientNet1 LinkNet (this work)** |         83.93        |   92.88          |   91.33          |    85.35                | **92.09**                  |
| **EfficientNet2 U-Net (this work)**   |         84.09        |   92.23          |   91.57          |    85.01                | 91.90                      |
| **Ensemble U-Net LinkNet (this work)**|         **84.42**    |   92.68          |  **91.80**       |    **85.51**            | 92.07                      |
