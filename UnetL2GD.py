import numpy as np
import glob
import h5py
import math
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose, concatenate, Reshape, Dropout, Add, Lambda
from keras.models import Model, load_model
from keras.utils import np_utils, to_categorical
from keras import callbacks
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator 
from keras import regularizers
from keras import metrics
import keras.backend as K
from keras.utils import plot_model
import tensorflow as tf
from scipy import ndimage
np.random.seed(12345)  # for reproducibility


####### Loss Function #######
## Github Reference: https://github.com/ginobilinie/medSynthesis
def L2GD(y_true, y_pred):
	## L2(mean squared error) function ##
	L2 = tf.reduce_sum(tf.abs(y_pred - y_true)**2)

    	# create filters (kernels) [-1, 1] and [[1],[-1]] for gradient in x and y direction 
    	pos = tf.constant(np.identity(1), dtype=tf.float32)
    	neg = -1 * pos
    	G_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    	G_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    	strides = [1, 1, 1, 1]  # stride of 1  	
	padding = 'SAME'
	
	## convolution of kernels(G_x or G_y) with Output from network and Label
    	pred_dx = tf.abs(tf.nn.conv2d(y_pred, G_x, strides, padding=padding)) 
    	pred_dy = tf.abs(tf.nn.conv2d(y_pred, G_y, strides, padding=padding))
    	label_dx = tf.abs(tf.nn.conv2d(y_true, G_x, strides, padding=padding))
    	label_dy = tf.abs(tf.nn.conv2d(y_true, G_y, strides, padding=padding))
	
	## Difference between label and pred gradient in x or y
    	G_diffx = tf.abs(label_dx - pred_dx)
    	G_diffy = tf.abs(label_dy - pred_dy)
	# Gradient Difference Loss(GDL) = Sum of losses in x and y
	GDL=tf.reduce_sum(G_diffx + G_diffy)

	return L2 + GDL

###################################### MODEL DEFINATION START ######################################
my_Adam = Adam(lr=0.0001)
## Input data dimensions ##
width = 256
height = 256
inputs = Input(shape=(height,width,6)) 
fracDO = 0.2
kerReg = 0.0
## Stage 1 ##
C1_1        = Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
C1_1        = BatchNormalization(center=False, scale=False)(C1_1)
C1_1        = Activation('relu')(C1_1)
C1_2        = Conv2D(64, kernel_size=3, strides=1, padding='same')(C1_1)
C1_2        = BatchNormalization(center=False, scale=False)(C1_2)
C1_2        = Activation('relu')(C1_2)
C1_3        = Conv2D(64, kernel_size=3, strides=1, padding='same')(C1_2)
C1_3        = BatchNormalization(center=False, scale=False)(C1_3)
C1_3        = Activation('relu')(C1_3)
C1_pool     = MaxPooling2D(pool_size=(2, 2), strides=2)(C1_3)

## Stage 2 ##
C2_1        = Conv2D(128, kernel_size=3, strides=1, padding='same')(C1_pool)
C2_1        = BatchNormalization(center=False, scale=False)(C2_1)
C2_1        = Activation('relu')(C2_1)
C2_2        = Conv2D(128, kernel_size=3, strides=1, padding='same')(C2_1)
C2_2        = BatchNormalization(center=False, scale=False)(C2_2)
C2_2        = Activation('relu')(C2_2)
C2_3        = Conv2D(128, kernel_size=3, strides=1, padding='same')(C2_2)
C2_3        = BatchNormalization(center=False, scale=False)(C2_3)
C2_3        = Activation('relu')(C2_3)
C2_pool     = MaxPooling2D(pool_size=(2, 2), strides=2)(C2_3)

## Stage 3 ##
C3_1        = Conv2D(256, kernel_size=3, strides=1, padding='same')(C2_pool)
C3_1        = BatchNormalization(center=False, scale=False)(C3_1)
C3_1        = Activation('relu')(C3_1)
C3_2        = Conv2D(256, kernel_size=3, strides=1, padding='same')(C3_1)
C3_2        = BatchNormalization(center=False, scale=False)(C3_2)
C3_2        = Activation('relu')(C3_2)
C3_3        = Conv2D(256, kernel_size=3, strides=1, padding='same')(C3_2)
C3_3        = BatchNormalization(center=False, scale=False)(C3_3)
C3_3        = Activation('relu')(C3_3)
C3_pool     = MaxPooling2D(pool_size=(2, 2), strides=2)(C3_3)

## Stage 4 ##
C4_1        = Conv2D(512, kernel_size=3, strides=1, padding='same')(C3_pool)
C4_1        = BatchNormalization(center=False, scale=False)(C4_1)
C4_1        = Activation('relu')(C4_1)
C4_2        = Conv2D(512, kernel_size=3, strides=1, padding='same')(C4_1)
C4_2        = BatchNormalization(center=False, scale=False)(C4_2)
C4_2        = Activation('relu')(C4_2)
C4_3        = Conv2D(512, kernel_size=3, strides=1, padding='same')(C4_2)
C4_3        = BatchNormalization(center=False, scale=False)(C4_3)
C4_3        = Activation('relu')(C4_3)
C4_pool     = MaxPooling2D(pool_size=(2, 2), strides=2)(C4_3)

## Stage 5 ##
C5_1        = Conv2D(1024, kernel_size=3, strides=1, padding='same')(C4_pool)
C5_1        = BatchNormalization(center=False, scale=False)(C5_1)
C5_1        = Activation('relu')(C5_1)
C5_2        = Conv2D(1024, kernel_size=3, strides=1, padding='same')(C5_1)
C5_2        = BatchNormalization(center=False, scale=False)(C5_2)
C5_2        = Activation('relu')(C5_2)
C5_3        = Conv2D(1024, kernel_size=3, strides=1, padding='same')(C5_2)
C5_3        = BatchNormalization(center=False, scale=False)(C5_3)
C5_3        = Activation('relu')(C5_3)



## Stage 6 Decoder ##
C4D_UpSamp      = UpSampling2D(size=(2, 2))(C5_3)
C4D_Conct       = concatenate([C4D_UpSamp, C4_3], axis=-1)
C4D_1           = Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C4D_Conct)
C4D_1           = BatchNormalization(center=False, scale=False)(C4D_1)
C4D_1           = Activation('relu')(C4D_1)
C4D_2           = Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C4D_1)
C4D_2           = BatchNormalization(center=False, scale=False)(C4D_2)
C4D_2           = Activation('relu')(C4D_2)

## Stage 7 Decoder ##
C3D_UpSamp      = UpSampling2D(size=(2, 2))(C4D_2)
C3D_Conct       = concatenate([C3D_UpSamp, C3_3], axis=-1)
C3D_1           = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C3D_Conct)
C3D_1           = BatchNormalization(center=False, scale=False)(C3D_1)
C3D_1           = Activation('relu')(C3D_1)
C3D_2           = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C3D_1)
C3D_2           = BatchNormalization(center=False, scale=False)(C3D_2)
C3D_2           = Activation('relu')(C3D_2)

## Stage 8 Decoder ##
C2D_UpSamp      = UpSampling2D(size=(2, 2))(C3D_2)
C2D_Conct       = concatenate([C2D_UpSamp, C2_3], axis=-1)
C2D_1           = Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C2D_Conct)
C2D_1           = BatchNormalization(center=False, scale=False)(C2D_1)
C2D_1           = Activation('relu')(C2D_1)
C2D_2           = Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C2D_1)
C2D_2           = BatchNormalization(center=False, scale=False)(C2D_2)
C2D_2           = Activation('relu')(C2D_2)

## Stage 9 Decoder ##
C1D_UpSamp      = UpSampling2D(size=(2, 2))(C2D_2)
C1D_Conct       = concatenate([C1D_UpSamp, C1_3], axis=-1)
C1D_1           = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C1D_Conct)
C1D_1           = BatchNormalization(center=False, scale=False)(C1D_1)
C1D_1           = Activation('relu')(C1D_1)
C1D_2           = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C1D_1)
C1D_2           = BatchNormalization(center=False, scale=False)(C1D_2)
C1D_2           = Activation('relu')(C1D_2)

## Stage 9 Decoder ##
C00D_1          = Conv2D(1, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(C1D_2)

model           = Model(inputs=inputs, outputs=C00D_1)

model.compile(optimizer=my_Adam, loss=L2GD,metrics=['mean_squared_error'])
## Plot of Model ##
plot_model(model, to_file='/home/ajfer6/ab57/anthony-PET/MRIH.png',show_shapes=True)
print ('Model defination Complete !!!')
###################################### MODEL DEFINATION END ######################################
 
####### DATA GENERATOR using file located in pathVal and pathTrain#######
def generate_trainData(path, batchSize):
    while 1:
        fname = glob.glob(path)
        numFile = len(fname)
        assert(numFile>0), "No files in the folder"
        indxShuffle = np.random.permutation(numFile)
   	for n in indxShuffle:
            fh5 = h5py.File(fname[n],'r')
            data = np.array(fh5["data"])
            dataImg = np.array(fh5["labelImg"])
            SLC = data.shape[0]
	    c = 0
            for nBatch in range(0,SLC-7,2):
                x = np.transpose(data[nBatch:nBatch+batchSize,:,:],(1,2,0))
                x = x[None, :, :,: ]
		x = np.float32(x)
                y = np.transpose(dataImg[c:c+1,:,:],(1,2,0))
		c = c+1
	        y = y[None, :, :,: ]
		y = np.float32(y)
                yield x, y


def generate_valData(path, batchSize):
    while 1:
        fname = glob.glob(path)
        numFile = len(fname)
        assert(numFile>0), "No files in the folder"
	for n in range(0,numFile):
            fh5 = h5py.File(fname[n],'r')
            data = np.array(fh5["data"])
            dataImg = np.array(fh5["labelImg"])
            SLC = data.shape[0]
	    c = 0
            for nBatch in range(0,SLC-7,2):
                x = np.transpose(data[nBatch:nBatch+batchSize,:,:],(1,2,0))
                x = x[None, :, :,: ]
		x = np.float32(x)
                y = np.transpose(dataImg[c:c+1,:,:],(1,2,0))
		c = c+1
	        y = y[None, :, :,: ]
		y = np.float32(y)
                yield x, y


####### deinfe paths ########
pathTrain = '/home/ajfer6/ab57/anthony-PET/FinalTests/TrainPETN3/*.h5'
pathVal   = '/home/ajfer6/ab57/anthony-PET/FinalTests/ValidatePETN3/*.h5'
filepath  = "/home/ajfer6/ab57/anthony-PET/FinalTests/TestUnetL2GDF/model-{epoch:03d}.hdf5"
logpath   = '/home/ajfer6/ab57/anthony-PET/FinalTests/UnetL2GDF/'

### Define callbacks
## Logfile on tensorboard that saves training and validation loss and metrics
TfBoardCb = callbacks.TensorBoard(log_dir=logpath, histogram_freq=0)
## Generator function call##
mygenerator = generate_trainData(pathTrain, 6)
mygeneratorVal = generate_valData(pathVal, batchSize=6)
## saves weights of model to filepath#
modelsave = callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=True, period=1)
callbacks_list = [TfBoardCb, modelsave]

### Train
print ('Training Started')
## fit generator used to training network ##
model.fit_generator(mygenerator, steps_per_epoch=500, epochs=250, verbose=1, callbacks=callbacks_list, validation_data=mygeneratorVal, validation_steps=123, use_multiprocessing=True)
print ('Training Completed')

