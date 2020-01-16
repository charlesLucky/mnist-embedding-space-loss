## dataset
from tensorflow.keras.datasets import mnist

## for Model definition/training
from tensorflow.keras.models import Model, load_model,Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, models
from    tensorflow.keras import regularizers
from tripletloss import triplet_loss_adapted_from_tf,triplet_loss_plus_space_max_adapted_from_tf

## for visualizing 
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA
import os
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


parser = argparse.ArgumentParser(description='Split the data and generate the train and test set')
parser.add_argument('BATCH_SIZE', help='the BATCH_SIZE', nargs='?',default=128, type=int)

args = parser.parse_args()
BATCH_SIZE = args.BATCH_SIZE

def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def create_base_network(input_shape, embedding_size):
    """
    Base network to be shared (eq. to feature extraction).
    """

    weight_decay = 0.000

    model = Sequential()

    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(embedding_size))
    # model.add(layers.Activation('softmax'))
    return model



epochs = 10
train_flag = True  # either     True or False

embedding_size = 64
no_of_components = 2  # for visualization -> PCA.fit_transform()
step = 10

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
input_image_shape = (28, 28, 1)
x_val = x_test[:2000, :, :]
y_val = y_test[:2000]

IMG_WIDTH, IMG_HEIGHT, channels = 28, 28, 1

base_network = create_base_network(input_image_shape, embedding_size)

input_images = Input(shape=input_image_shape, name='input_image')  # input layer for images
input_labels = Input(shape=(1,), name='input_label')  # input layer for labels
embeddings = base_network([input_images])  # output of network -> embeddings
labels_plus_embeddings = concatenate([input_labels, embeddings])  # concatenating the labels + embeddings

# Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
model = Model(inputs=[input_images, input_labels],
              outputs=labels_plus_embeddings)

model.summary()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# train session
opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!

model.compile(loss=triplet_loss_adapted_from_tf,
              optimizer=opt)

filepath = "semiH_trip_MNIST_v13_ep{epoch:02d}_BS%d.hdf5" % BATCH_SIZE
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=25)
callbacks_list = [checkpoint]

# Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
dummy_gt_train = np.zeros((len(x_train), embedding_size + 1))
dummy_gt_val = np.zeros((len(x_val), embedding_size + 1))

x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], x_train.shape[1], 1))
x_val = np.reshape(x_val, (len(x_val), x_train.shape[1], x_train.shape[1], 1))

H = model.fit(
    x=[x_train, y_train],
    y=dummy_gt_train,
    batch_size=BATCH_SIZE,
    epochs=epochs,
    validation_data=([x_val, y_val], dummy_gt_val),
    callbacks=callbacks_list)

plt.figure(figsize=(8, 8))
plt.plot(H.history['loss'], label='training loss')
plt.plot(H.history['val_loss'], label='validation loss')
plt.legend()
plt.title('Train/validation loss')
#plt.show()
plt.savefig('trainlogs_tripletloss.png')

# Test the network

# creating an empty network
testing_embeddings = create_base_network(input_image_shape,
                                         embedding_size=embedding_size)
x_embeddings_before_train = testing_embeddings.predict(np.reshape(x_test, (len(x_test), 28, 28, 1)))
# Grabbing the weights from the trained network
for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[2].layers):
    weights = layer_source.get_weights()
    layer_target.set_weights(weights)
    del weights

x_embeddings = testing_embeddings.predict(np.reshape(x_test, (len(x_test), 28, 28, 1)))
dict_embeddings = {}
dict_gray = {}
test_class_labels = np.unique(np.array(y_test))
pca = PCA(n_components=no_of_components)
decomposed_embeddings = pca.fit_transform(x_embeddings)
    #     x_test_reshaped = np.reshape(x_test, (len(x_test), 28 * 28))
decomposed_gray = pca.fit_transform(x_embeddings_before_train)

fig = plt.figure(figsize=(16, 8))
for label in test_class_labels:
        decomposed_embeddings_class = decomposed_embeddings[y_test == label]
        decomposed_gray_class = decomposed_gray[y_test == label]

        plt.subplot(1, 2, 1)
        plt.scatter(decomposed_gray_class[::step, 1], decomposed_gray_class[::step, 0], label=str(label))
        plt.title('before training (embeddings)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(decomposed_embeddings_class[::step, 1], decomposed_embeddings_class[::step, 0], label=str(label))
        plt.title('after @%d epochs' % epochs)
        plt.legend()

plt.show()
