# ================================================================================
# Import modules
# ================================================================================
import sys
import os, random
import numpy as np
import tensorflow as tf 
from sklearn.cross_validation import train_test_split
from PIL import Image
from tqdm import tqdm
from keras.utils import np_utils
from keras.layers import Input, merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================================
# Model Architecture
# ================================================================================

##### Generative Model

# Constants
NUM_SIZE = 10
FILTERS = 1024
G_LR = 2e-4
D_LR = 1e-5


# ================================================================================
# Set up Data
# ================================================================================
img_rows, img_cols = 64, 64

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# ================================================================================
# Being catdog data addition
# ================================================================================
path2 = "/home/ubuntu/catdog/courses/deeplearning1/nbs/data/preprocessed"

imlist = os.listdir(path2)

imlist = np.sort(imlist).tolist()

immatrix = np.array(
    [np.array(Image.open(path2 + '/' + im2)).flatten()
              for im2 in imlist],
    'f')

num_samples = np.size(imlist) # 100

label=np.ones((num_samples,),dtype = int)
# currently all dogs
#label[0:50]=0   # cats
#label[51:100]=1 # dogs

from sklearn.utils import shuffle
data,label = shuffle(immatrix,label, random_state=2)
train_data = [data,label]

# Separate data into images and labels
(X, y) = (train_data[0],train_data[1])

# Separate data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Resize X
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print "X_train shape:", X_train.shape
print "X_test shape:", X_test.shape
print "y_train shape:", y_train.shape
print "y_test shape:", y_test.shape

# ================================================================================
# End my data addition
# ================================================================================


print np.min(X_train), np.max(X_train)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

shp = X_train.shape[1:]
print shp

g_input = Input(shape=[NUM_SIZE])
G = Dense(FILTERS * 4 * 4)(g_input)
G = BatchNormalization(mode=2)(G)
G = Activation('relu')(G)
G = Reshape((4, 4, FILTERS))(G)
G = UpSampling2D(size=(2,2))(G)
G = Convolution2D(FILTERS/2, 3,3, subsample=(1,1), border_mode='same')(G)
#G = Conv2DTranspose(FILTERS/2, (5,5), strides=(2,2))(G)
G = BatchNormalization(mode=2)(G)
G = Activation('relu')(G)
G = UpSampling2D(size=(2,2))(G)
G = Convolution2D(FILTERS/4, 3,3, subsample=(1,1), border_mode='same')(G)
#G = Conv2DTranspose(FILTERS/4, (5,5), strides=(2,2))(G)
G = BatchNormalization(mode=2)(G)
G = Activation('relu')(G)
G = UpSampling2D(size=(2,2))(G)
G = Convolution2D(FILTERS/8, 3,3, subsample=(1,1), border_mode='same')(G)
#G = Conv2DTranspose(FILTERS/8, (5,5), strides=(2,2))(G)
G = BatchNormalization(mode=2)(G)
G = Activation('relu')(G)
G = UpSampling2D(size=(2,2))(G)
G = Convolution2D(3, 3,3, subsample=(1,1), border_mode='same')(G)
g_V = Activation('tanh')(G)

generator = Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=G_LR, beta_1=0.5))
# print m.summary()

d_input = Input(shape=[64, 64, 3])
D = Convolution2D(FILTERS/16, 5,5, subsample=(2,2), border_mode='same')(d_input)
D = BatchNormalization(mode=2)(D)
D = LeakyReLU()(D)
D = Convolution2D(FILTERS/8, 5,5, subsample=(2,2), border_mode='same')(D)
D = BatchNormalization(mode=2)(D)
D = LeakyReLU()(D)
D = Convolution2D(FILTERS/4, 5,5, subsample=(2,2), border_mode='same')(D)
D = BatchNormalization(mode=2)(D)
D = LeakyReLU()(D)
D = Convolution2D(FILTERS/2, 3,3, subsample=(2,2), border_mode='same')(D)
D = LeakyReLU()(D)
D = BatchNormalization(mode=2)(D)
D = Flatten()(D) 
d_V = Dense(2, activation='softmax')(D)

discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(lr=D_LR, beta_1=0.5))
# print discriminator.summary()

# ================================================================================
# Function to freeze weights in the discriminator for stacked training
# ================================================================================
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discriminator, False)

# ================================================================================
# Build stacked GAN model
# ================================================================================
gan_input = Input(shape=[NUM_SIZE])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=Adam(lr=G_LR, beta_1=0.5))
print GAN.summary()


# ================================================================================
# Plotting functions to visualize images and loss
# ================================================================================
def plot_loss(losses, i):
    plt.figure(figsize=(10,8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()
    plt.savefig('plots/lossplot{}.png'.format(i))
    plt.close('all')

def plot_gen(num, n_ex=16,dim=(4,4), figsize=(10,10)):
    noise = np.random.uniform(0,1,size=[n_ex,NUM_SIZE])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/generatedimage{}.png'.format(num))
    plt.close('all')

# ================================================================================
# Pre-train discriminator network
# ================================================================================
# ntrain = 100
# trainidx = random.sample(range(0,X_train.shape[0]), ntrain)
# XT = X_train[trainidx,:,:,:]
# 
# # Pre-train the discriminator network 
# noise_gen = np.random.uniform(0,1,size=[XT.shape[0],NUM_SIZE])
# generated_images = generator.predict(noise_gen)
# 
# print("generated images shape:",  generated_images.shape)
# print("XT shape: ", XT.shape)
# 
# 
# X = np.concatenate((XT, generated_images))
# n = XT.shape[0]
# y = np.zeros([2*n,2])
# y[:n,1] = 1
# y[n:,0] = 1
# 
# make_trainable(discriminator,True)
# discriminator.fit(X,y, nb_epoch=1, batch_size=32)
# y_hat = discriminator.predict(X)
# 
# 
# y_hat_idx = np.argmax(y_hat,axis=1)
# y_idx = np.argmax(y,axis=1)
# diff = y_idx-y_hat_idx
# n_tot = y.shape[0]
# n_rig = (diff==0).sum()
# acc = n_rig*100.0/n_tot
# print "Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot)



# set up loss storage vector
losses = {"d":[], "g":[]}


# ================================================================================
# Define training function
# ================================================================================
def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    r e in tqdm(range(nb_epoch)):
  
        print "epoch number: {}".format(e)
        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]    
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,NUM_SIZE])
        generated_images = generator.predict(noise_gen, verbose=2)fo
        
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(discriminator,True)
        make_trainable(generator,False)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,NUM_SIZE])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        make_trainable(discriminator,False)
        make_trainable(generator,True)
        g_loss = GAN.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)
        
        print "generator loss:{}, discriminator loss:{}".format(g_loss,d_loss)

        # Updates plots
        if e%plt_frq==plt_frq-1:
            plot_loss(losses, 'loss')
            plot_gen(num=e)


train_for_n(nb_epoch=4000, plt_frq=10,BATCH_SIZE=32)
