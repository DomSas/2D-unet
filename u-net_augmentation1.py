import tensorflow
import numpy as np
import segmentation_models as sm
import matplotlib.pyplot as plt
import argparse
import albumentations as A
#import cv2
import os

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import binary_crossentropy, DiceLoss
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from skimage.transform import resize
from skimage.io import imsave

#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from ImageDataAugmentor.image_data_augmentor import *

sm.set_framework('tf.keras')

SEED = 1
N_CLASSES = 2

img_rows = int(192)
img_cols = int(192)
smooth = 1.

BACKBONE = 'resnet34'
METRICS = DiceLoss
LOSS = 'binary_crossentropy'

preprocess_input = get_preprocessing(BACKBONE)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def load_train_data():
    imgs_train = np.load('../data40_npy_loaded_WithVal/40_imgs_train.npy')
    masks_train = np.load('../data40_npy_loaded_WithVal/40_masks_train.npy')
    return imgs_train, masks_train

def load_val_data():
    imgs_val = np.load('../data40_npy_loaded_WithVal/40_imgs_val.npy')
    masks_val = np.load('../data40_npy_loaded_WithVal/40_masks_val.npy')
    return imgs_val, masks_val

def load_test_data():
    imgs_test = np.load('../data40_npy_loaded_WithVal/40_imgs_test.npy')
    return imgs_test


print('-'*20)
print('Loading the data')
print('-'*20)

imgs_train, masks_train = load_train_data()
imgs_val, masks_val = load_val_data()

imgs_test = load_test_data()

##################################### Preprocess data #####################################
# all data are already normalized in range [0,1] and float32

# Train
imgs_train = preprocess(imgs_train)
imgs_train = preprocess_input(imgs_train)

masks_train = preprocess(masks_train)
masks_train = preprocess_input(masks_train)

# Val
imgs_val = preprocess(imgs_val)
imgs_val = preprocess_input(imgs_val)

masks_val = preprocess(masks_val)
masks_val = preprocess_input(masks_val)

# Test
imgs_test = preprocess(imgs_test)
imgs_test = preprocess_input(imgs_test)


albumentation_combo = A.Compose([
    A.ShiftScaleRotate(p=1)
    ])

def one_hot_encode_masks(y:np.array, classes=range(N_CLASSES)):
    ''' One hot encodes target masks for segmentation '''
    y = y.squeeze()
    masks = [(y == v) for v in classes]
    mask = np.stack(masks, axis=-1).astype('float')
    # add background if the mask is not binary
    if mask.shape[-1] != 1:
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)
    return mask

##################################### Augmentation of training data #####################################

img_data_gen = ImageDataAugmentor(
    augment=albumentation_combo,
    input_augment_mode='image',
    validation_split=0.2,
    seed=SEED,
)

mask_data_gen = ImageDataAugmentor(
    augment=albumentation_combo, 
    input_augment_mode='mask', #<- notice the different augment mode
#    preprocess_input=one_hot_encode_masks,
    validation_split=0.2,
    seed=SEED,
)

tr_img_gen = img_data_gen.flow(imgs_train, batch_size=32, subset='training')
tr_mask_gen = mask_data_gen.flow(masks_train, batch_size=32, subset='training')

##################################### Augmentation of validation data #####################################

val_imgs_gen = img_data_gen.flow(imgs_val, batch_size=32, subset='validation')
val_masks_gen = mask_data_gen.flow(masks_val, batch_size=32, subset='validation')

train_generator = zip(tr_img_gen, tr_mask_gen)
validation_generator = zip(tr_img_gen, tr_mask_gen)


##################################### Define the model #####################################

N = imgs_train.shape[-1] # number of channels

base_model = Unet(BACKBONE, encoder_weights='imagenet')

inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)


##################################### Compile the model #####################################

model.compile('Adam', loss=DiceLoss(), metrics=[dice_coef])


print('-'*20)
print('Training the model')
print('-'*20)

my_callbacks = [ModelCheckpoint('40_augment_weights.h5', monitor='val_loss', save_best_only=True)]


model.fit(
  train_generator,
  steps_per_epoch=180, # number of slices (192*30 = 6720) / batch size (32)
  epochs=50,
  validation_data=validation_generator,
  validation_steps=30,
  callbacks=[my_callbacks]
)



print('-'*20)
print('Predicting')
print('-'*20)

model.load_weights('40_augment_weights.h5')

imgs_mask_test = model.predict(imgs_test, verbose=1)
np.save('40_augment_predicted.npy', imgs_mask_test)


plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Model dice coeff')
plt.ylabel('Dice coeff')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('40_augment_results.png')


print('-'*20)
print("Finished")
