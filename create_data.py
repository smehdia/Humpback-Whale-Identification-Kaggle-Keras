# import necessary packages
import cv2
import pandas as pd
import numpy as np
import pickle
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

    # read csv of training data
    whales_data = pd.read_csv(CSV_PATH)
    # extract labels (which are names of the whales)
    labels = list(whales_data['Id'])
    integer_labels = range(len(list(set(labels))))
    labels_integers_map = dict(zip(list(set(labels)), integer_labels))
    # get image name
    images_name = list(whales_data['Image'])

    # read images and create dataset
    images = []
    labels_integers = []
    for i in range(len(labels)):
        # convert labels to onehot vector
        label_name = labels[i]
        label_integer = labels_integers_map[label_name]
        image_id = images_name[i]
        filename = IMAGES_PATH + '/' + image_id
        image = cv2.imread(filename, 0)
        image = preprocess(image)
        images.append(image)
        labels_integers.append(label_integer)
        print(filename)
        print('Number of Enteries: {}'.format(i))
        print('---------------------------------')

    print('Saving Data ...')
    # save data as numpy arrays
    np.save('data/labels.npy', np.array(labels_integers))
    np.save('data/images.npy', np.array(images))
    with open('data/'+ 'name_to_integer_map' + '.pkl', 'wb') as f:
        pickle.dump(labels_integers_map, f, pickle.HIGHEST_PROTOCOL)
    print('Processing Completed')