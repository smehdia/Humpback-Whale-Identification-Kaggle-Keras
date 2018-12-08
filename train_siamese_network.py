from keras.models import Model
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Activation, Dropout
from keras import regularizers
from keras.optimizers import RMSprop
import keras.backend as K
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt

'''
define constant parameters and hyper parameters here
'''
# general parameters
IMG_WIDTH = 32
IMG_HEIGHT = 16
NUM_CHANNELS = 1
# network parameters
NUM_BASE_FILTERS = 16
REGULARIZATION_PARAM = 1e-3
DROPOUT = 0.25
ALPHA_LEAKY_RELU = 0.2
NUM_DENSE = 64
# training options
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_ITERATIONS = 100000
MAX_EPOCHS = 1000
# remove logs from previous runs
try:
    shutil.rmtree('logs')
except:
    pass


# cnn model definition in keras
def build_siamese_model(num_base_filters, alpha_leaky_relu, rg, num_elements_of_vector):
    
    input_image = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    x = Conv2D(filters=num_base_filters, kernel_size=(5, 5), kernel_regularizer=regularizers.l2(rg),
               name='first_layer')(input_image)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)
    #x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=2 * num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(rg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)
    #x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=4 * num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(rg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)
    #x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=8 * num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(rg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)

    x = Flatten()(x)

    x = Dense(num_elements_of_vector)(x)
    x = Activation('sigmoid')(x)

    # build API model in keras
    conv_model = Model(inputs=[input_image], outputs=[x])

    input_image_siamese1 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    input_image_siamese2 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    encoded_image1 = conv_model(input_image_siamese1)
    encoded_image2 = conv_model(input_image_siamese2)
    # merge two encoded inputs with the l1 distance between them
    L1_distance = Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_image1, encoded_image2])
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    siamese_net = Model(input=[input_image_siamese1, input_image_siamese2], output=prediction)

    siamese_net.summary()

    return siamese_net

def prepare_data():
    X = np.load('data/images.npy')
    y = np.load('data/labels.npy')
    # split data to train and test (use some of data for test which have more samples than others)
    test_indices = []
    for i in range(len(set(y.tolist()))):
        number_of_enteries = y[np.where(y == i)].shape[0]
        if number_of_enteries > 4:
            test_indices += list(np.random.choice(np.where(y==i)[0], 2))

    train_indices = list(set(range(X.shape[0])) ^ set(test_indices))
    X_test, y_test = X[test_indices], y[test_indices]
    X_train, y_train = X[train_indices], y[train_indices]

    return X_train, y_train, X_test, y_test

def prepare_batch(X, y, indices_for_training):
    X_batch, y_batch = [], []
    for i in range(BATCH_SIZE // 2):
        selected_index = random.choice(indices_for_training)
        indices_for_training.remove(selected_index)
        whale_id = y[selected_index]
        same_whale_index = np.random.choice(np.where(y == whale_id)[0], 1)
        X_batch += [[X[selected_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1])/255., X[same_whale_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1])/255.]]
        diff_whale_index = np.random.choice(np.where(y != whale_id)[0], 1)
        X_batch += [[X[selected_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1])/255., X[diff_whale_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1])/255.]]
        y_batch += [1, 0]

    return np.array(X_batch), np.array(y_batch), indices_for_training

if __name__ == "__main__":

    siamese_net = build_siamese_model(num_base_filters=16, alpha_leaky_relu=0.1, rg=1e-4, num_elements_of_vector=1000)
    rms = RMSprop(LEARNING_RATE)
    siamese_net.compile(loss='binary_crossentropy', optimizer=rms, metrics=['mse'])

    X_train, y_train, X_test, y_test = prepare_data()

    total_binary_loss = []
    total_mse_loss = []
    for epoch in range(MAX_EPOCHS):
        indices_for_training = range(X_train.shape[0])
        for iteration in range(MAX_ITERATIONS):
            if len(indices_for_training) > BATCH_SIZE:
                X_batch, y_batch, indices_for_training  = prepare_batch(X_train, y_train, indices_for_training)

                binary_loss, mse = siamese_net.train_on_batch([X_batch[:,0], X_batch[:,1]], y_batch)
                total_binary_loss.extend([binary_loss])
                total_mse_loss.extend([total_mse_loss])
                print binary_loss
                plt.plot(total_binary_loss)
                plt.savefig('loss.png')
                plt.close()














































