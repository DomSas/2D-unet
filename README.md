# 2D-unet project for master thesis

## Contents of each file:

### u-net_augmentation1.py
Contains code to train with generator that generates augmented samples with Albumentations library. Code is based on <https://github.com/mjkvaak/ImageDataAugmentor>

### data_preprocess2021mar.ipynb
Loading and preprocessing input data. This file includes also 2D U-Net model, which is there just to predict images with already pretrained weights. After this prediction, slices of predicted images can be saved.

### server_code_model2.py
This file contains code to be run on server with python3.

### server_code_model_mar_2D_nonzero.py
This file contains code to be run on server with python3. It loads datasets which do not contain slices without any lesions in their mask. 
