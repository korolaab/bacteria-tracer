import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input,MaxPooling2D, Conv2D,Dropout,Conv2DTranspose, concatenate,LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
import random
import math
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
dtype="float32"
def augmentation(x,y,batch_size=32):#augmentation
    size = x.shape[1]
    x_train = np.empty([batch_size,size,size,x.shape[3]],dtype="float32")
    y_train = np.empty([batch_size,size,size,y.shape[3]],dtype="float32")
    while True:
        indexes = random.sample(range(0, x.shape[0]-1), batch_size)
        for i in range(batch_size):
            flipl= bool(random.getrandbits(1))
            flipr=bool(random.getrandbits(1))
            gamma = random.uniform(0.5,1.5)
            # flips and transpose
            x_temp = x[i,:,:,:]
            y_temp = y[i,:,:,:]

            if(flipl):
                x_temp = x_temp[:,::-1]
                y_temp = y_temp[:,::-1]
            if(flipr):
                x_temp = np.flipud(x_temp)
                y_temp = np.flipud(y_temp)
            #scale gamma
            x_temp = adjust_gamma(x_temp,gamma)
            x_train[i,:,:,:]=x_temp
            y_train[i,:,:,:]=y_temp
        yield x_train,y_train


def adjust_gamma(image, gamma=1.0):
	image = image**gamma
	return image

def crop_from_photo(image,size,shift,mode=None):# give pieces of images  
     N = math.ceil(image.shape[0]/shift)*math.ceil(image.shape[1]/shift)
     pieces = np.zeros([N,size,size,image.shape[2]],dtype=dtype)
     n=0
     for x_shift in range(math.ceil(image.shape[0]/shift)):
         for y_shift in range(math.ceil(image.shape[1]/shift)):
             x = shift*x_shift
             y = shift*y_shift
             piece = image[x:x+size,y:y+size,:]
             pieces[n,:piece.shape[0],:piece.shape[1],:]=piece
             n+=1
     return pieces

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=False):
    dr=1 # dilation_rate
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same",dilation_rate=(dr,dr),use_bias=True)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x =LeakyReLU(alpha=0)(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same",dilation_rate=(dr,dr),use_bias=True)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0)(x)
    return x

def focal_loss(y_true, y_pred): #https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
        gamma = 2.0
        alpha=0.25
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def graph(input_shape = (100,100,1),n_filters=8,batchnorm=False,dropout=0.3,
                    pretrained_weights=None):
    # n_filters - number of conv2D kernels
    inputs = Input(input_shape,dtype="float32")
    c1 = conv2d_block(inputs, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(c1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u5 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c4)
    u5 = concatenate([u5, c3])
    u5 = Dropout(dropout)(u5)
    c5 = conv2d_block(u5, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)


    u6 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c2])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)


    out = Conv2D(1, (1, 1),activation="sigmoid") (c6)

    model = Model(input = inputs, output = out)

    model.compile(optimizer=Adam(lr=1e-3), loss=focal_loss)
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
