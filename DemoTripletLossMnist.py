import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, models
from tensorflow.keras.layers import  Input, Flatten, Dense, concatenate,  Dropout
from    tensorflow.keras import regularizers
import pathlib
import datetime
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tripletloss import triplet_loss_adapted_from_tf
import argparse
from tensorflow.keras.datasets import mnist
import os
import numpy as np
parser = argparse.ArgumentParser(description='Split the data and generate the train and test set')
parser.add_argument('BATCH_SIZE', help='the BATCH_SIZE', nargs='?',default=32, type=int)

args = parser.parse_args()
BATCH_SIZE = args.BATCH_SIZE

def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def create_base_network(image_input_shape, embedding_size):
    """
    Base network to be shared (eq. to feature extraction).
    """
    input_image = Input(shape=image_input_shape)

    x = Flatten()(input_image)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(embedding_size)(x)

    base_network = Model(inputs=input_image, outputs=x)
    plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
    return base_network


epochs = 25
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


weight_decay = 0.000

model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_image_shape,
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

#model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())

timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


input_images = Input(shape=input_image_shape, name='input_image')  # input layer for images
input_labels = Input(shape=(1,), name='input_label')  # input layer for labels
embeddings = model(input_images)  # output of network -> embeddings
labels_plus_embeddings = concatenate([input_labels, embeddings])  # concatenating the labels + embeddings

# Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
model = models.Model(inputs=[input_images, input_labels],
                  outputs=labels_plus_embeddings)

model.summary()

    # train session

model.compile(
        # loss=tfa.losses.TripletSemiHardLoss(),
        loss=triplet_loss_adapted_from_tf,
        optimizer=keras.optimizers.Adam(0.001),
        metrics=['accuracy'])

    # 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = "model/" + timestamp + "inceptionrenet_semiH_trip_v13_ep-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
logdir = os.path.join("logs/inceptionresnet_tripletloss_trainwithSTnLT_ID_SPLIT", timestamp)
check_folder(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    # This callback will stop the training when there is no improvement in
    # 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=10)

    # Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
dummy_gt_train = np.zeros((len(x_train), embedding_size + 1))
dummy_gt_val = np.zeros((len(x_val), embedding_size + 1))

x_train = np.reshape(x_train, (len(x_train),  IMG_WIDTH,IMG_HEIGHT, channels))
x_val = np.reshape(x_val, (len(x_val), IMG_WIDTH,IMG_HEIGHT,  channels))

history = model.fit(
        x=[x_train, y_train],
        y=dummy_gt_train,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        shuffle=True,
        callbacks=[cp_callback],
        validation_data=([x_val, y_val], dummy_gt_val),
        validation_steps=200)

