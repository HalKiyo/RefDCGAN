import os
from keras.datasets import mnist
from keras.optimizers import Adam
from PIL import Image
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU
from keras.layers import Flatten, Dropout

def generator_model():
    model = Sequential()
    model.add(Dense(input_shape=(100,), units=1024, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D((2, 2), data_format='channels_last'))
    model.add(Conv2D(64, (5, 5),
                     padding='same',
                     data_format='channels_last',
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2), data_format='channels_last'))
    model.add(Conv2D(1, (5, 5),
                    padding='same',
                    data_format='channels_last',
                    kernel_initializer='he_normal'))
    model.add(Activation('tanh'))

    return model

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5),
                    strides=(2, 2),
                    padding='same',
                    data_format='channels_last',
                    kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5, 5),
                    strides=(2, 2),
                    data_format='channels_last',
                    kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256,kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]

    return combined_image

def train():
    BATCH_SIZE = 32
    NUM_EPOCH = 20
    GENERATED_IMAGE_PATH = 'generated_images/'

    (X_train, y_train), (_,_) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    discriminator = discriminator_model()
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    # generator + discriminator (discriminator's weights are frozen)
    discriminator.trainable = False
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            # create imagery
            if index % 500 == 0:
                image = combine_images(generated_images)
                image = image*127.5 + 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8))\
                        .save(f"{GENERATED_IMAGE_PATH}{epoch}_{index}.png")

            # renew discriminator
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1]*BATCH_SIZE + [0]*BATCH_SIZE)
            d_loss = discriminator.train_on_batch(X, y)

            # renew generator
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, np.array([1]*BATCH_SIZE))
            print(f"epoch: {epoch}, batch: {index}, g_loss: {g_loss}, d_loss: {d_loss}")

        generator.save_weights('generator.h5')
        discriminator.save_weights('discriminator.h5')

if __name__ == '__main__':
    train()
