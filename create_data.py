# import necessary packages
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os

# directories of training data
IMAGES_PATH = '/home/mehdi/Desktop/Kaggle/Whale/train'
CSV_PATH = '/home/mehdi/Desktop/Kaggle/Whale/train.csv'
IMG_WIDTH = 128
IMG_HEIGHT = 64

if not os.path.isdir('data'):
    os.makedirs('data')


def preprocess(image):
    # resize image and refine contrast
    preprocessed_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_image = clahe.apply(preprocessed_image)

    return preprocessed_image


if __name__ == "__main__":

    # initialize encoders
    name_to_integet_encoder = LabelEncoder()
    integer_to_one_hot_encoder = OneHotEncoder()
    # read csv of training data
    whales_data = pd.read_csv(CSV_PATH)
    # extract labels (which are names of the whales)
    labels = list(whales_data['Id'])
    # get image name
    images_name = list(whales_data['Image'])
    # convert label name to integer
    name_to_integet_encoder.fit(labels)
    labels_integers = name_to_integet_encoder.transform(labels)
    # convert integers to one hot
    integer_to_one_hot_encoder.fit(np.array(labels_integers).reshape(-1, 1))
    labels_one_hot = integer_to_one_hot_encoder.transform(np.array(labels_integers).reshape([-1, 1])).toarray()

    # read images and create dataset
    images = []
    for i in range(len(labels)):
        label_one_hot = labels_one_hot[i]
        image_id = images_name[i]
        filename = IMAGES_PATH + '/' + image_id
        image = cv2.imread(filename, 0)
        image = preprocess(image)
        images.append(image)
        print(filename)
        print('Number of Enteries: {}'.format(i))
        print('---------------------------------')

    print('Saving Data ...')
    # save data as numpy arrays
    np.save('data/labels.npy', np.array(labels_one_hot))
    np.save('data/images.npy', np.array(images))
    print('Processing Completed')