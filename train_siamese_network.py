from keras.models import Model
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Activation, Dropout
from keras import regularizers
from keras.optimizers import RMSprop
import keras.backend as K
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from utils import TrainValTensorBoard
from keras.callbacks import Callback
import os
'''
define constant parameters and hyper parameters here
'''
# general parameters
IMG_WIDTH = 128
IMG_HEIGHT = 64
NUM_CHANNELS = 1
# network parameters
NUM_BASE_FILTERS = 36
REGULARIZATION_PARAM = 2e-4
DROPOUT = 0.25
ALPHA_LEAKY_RELU = 0.2
NUM_DENSE = 1000
# training options
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 25000
MAX_EPOCHS = 1000
# remove logs from previous runs
try:
    shutil.rmtree('logs')
    os.mkdir('models')
except:
    pass

# create custom callback in order to visualize first layer filters
class draw_first_layer_filters(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        # draw first layer filters
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        layer_dict_model1 = dict([(layer.name, layer) for layer in layer_dict['model_1'].layers])

        first_layer_weights = np.array(layer_dict_model1['first_layer'].get_weights()[0])
        sqrt_num_filters = int(np.ceil(np.sqrt(first_layer_weights.shape[3])))
        for i in range(sqrt_num_filters):
            for j in range(sqrt_num_filters):
                plt.subplot(sqrt_num_filters, sqrt_num_filters, j + sqrt_num_filters * i + 1)
                try:
                    plt.imshow(first_layer_weights[:, :, 0, j + sqrt_num_filters * i], cmap='gray',
                               interpolation='none')
                    plt.axis('off')
                except:
                    continue
        if epoch == 0:
            plt.savefig('first_layer_filters_before_training.png')
        else:
            plt.savefig('first_layer_filters.png')

        plt.close()


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

# cnn model definition in keras
def build_siamese_model(num_base_filters, alpha_leaky_relu, rg, num_elements_of_vector):
    
    input_image = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    x = Conv2D(filters=num_base_filters, kernel_size=(9, 9), kernel_regularizer=regularizers.l2(rg),
               name='first_layer')(input_image)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)
    x = MaxPooling2D((1, 2))(x)

    x = Conv2D(filters=2 * num_base_filters, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(rg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=4 * num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(rg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=8 * num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(rg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)

    x = Conv2D(filters=8 * num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(rg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha_leaky_relu)(x)

    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_elements_of_vector)(x)
    x = Activation('relu')(x)

    # build API model in keras
    conv_model = Model(inputs=[input_image], outputs=[x])

    input_image_siamese1 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    input_image_siamese2 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    encoded_image1 = conv_model(input_image_siamese1)
    encoded_image2 = conv_model(input_image_siamese2)
    # merge two encoded inputs with the l1 distance between them
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([encoded_image1, encoded_image2])

    siamese_net = Model(input=[input_image_siamese1, input_image_siamese2], output=distance)

    conv_model.summary()
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

import cv2
def prepare_batch(X, y, flag_test=False):
    X_first_input = []
    X_second_input = []
    y_batch = []
    if flag_test:
        num_data = X.shape[0]
    else:
        num_data = 25000

    for i in range(num_data):
        selected_index = random.choice(range(X.shape[0]-1))
        whale_id = y[selected_index]
        same_whale_index = np.random.choice(np.where(y == whale_id)[0], 1)
        X_first_input.append(X[selected_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1])/255.)
        X_second_input.append(X[same_whale_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1])/255.)
        X_first_input.append(X[selected_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1]) / 255.)
        diff_whale_index = np.random.choice(np.where(y != whale_id)[0], 1)
        X_second_input.append(X[diff_whale_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1]) / 255.)
        y_batch += [1, 0]

    return np.array(X_first_input), np.array(X_second_input), np.array(y_batch)

def prepare_test(X_train, X_test, y_train, y_test):
    img_test_batch = []
    img_class_batch = []
    for i in range(X_test.shape[0]):
        img_test = X_test[i].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1]) / 255.
        indices = np.where(y_train == y_test[i])[0]
        first_index = indices[0]
        img_class = X_train[first_index].astype('float32').reshape([IMG_HEIGHT, IMG_WIDTH, 1]) / 255.
        img_class_batch.append(img_class)
        img_test_batch.append(img_test)

    return np.array(img_test_batch), np.array(img_class_batch), np.ones(shape=X_test.shape[0])

if __name__ == "__main__":

    siamese_net = build_siamese_model(num_base_filters=NUM_BASE_FILTERS, alpha_leaky_relu=ALPHA_LEAKY_RELU, rg=REGULARIZATION_PARAM, num_elements_of_vector=NUM_DENSE)
    rms = RMSprop(LEARNING_RATE)
    siamese_net.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    try:
        siamese_net.load_weights('./models/model.h5')
    except:
        pass

    X_train, y_train, X_test, y_test = prepare_data()

    X_first_input_test, X_second_input_test, y_batch_test = prepare_batch(X_test, y_test)
    X_first_input, X_second_input, y_batch = prepare_batch(X_train, y_train)

    # define callbacks for saving model, early stopping, visualizing first layer filters
    checkpoint = ModelCheckpoint('./models/model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0, patience=50, verbose=0, mode='auto')
    draw_first_layer_filters_callback = draw_first_layer_filters()


    siamese_net.fit(x={'input_2':X_first_input, 'input_3':X_second_input}, y=y_batch, batch_size=BATCH_SIZE, epochs=500, verbose=1,
        validation_data=({'input_2':X_first_input_test, 'input_3':X_second_input_test}, y_batch_test), shuffle=True, callbacks=[checkpoint, early_stopping, draw_first_layer_filters_callback,TrainValTensorBoard(write_graph=False)])



























































