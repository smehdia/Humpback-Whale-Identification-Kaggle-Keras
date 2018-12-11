import cv2
import glob
import numpy as np
from keras.models import load_model
from train_siamese_network import contrastive_loss
import pickle
import time

IMG_WIDTH = 128
IMG_HEIGHT = 64

def preprocess(image):
    # resize image and refine contrast
    preprocessed_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_image = clahe.apply(preprocessed_image)

    return preprocessed_image

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":

    f = open('submission.csv', 'w')
    f.write('Image,ID\n')
    f.close()
    siamese_model = load_model('models/model.h5', custom_objects={"contrastive_loss": contrastive_loss})
    layer_dict = dict([(layer.name, layer) for layer in siamese_model.layers])
    layer_dict_model1 = dict([(layer.name, layer) for layer in layer_dict['model_1'].layers])
    encoder_model = siamese_model.layers[2]

    name_to_integer_map = load_obj('./data/name_to_integer_map')
    integer_to_name_map = {v: k for k, v in name_to_integer_map.iteritems()}
    X_training = np.load('data/images.npy')
    y_training = np.load('data/labels.npy')

    X_training = X_training.astype('float32') / 255
    X_training = np.expand_dims(X_training, axis=-1)

    encoded_training = np.array(encoder_model.predict(X_training))

    print 'Encoding Training Data Done'

    for i, filename in enumerate(glob.glob('/home/desktop/Desktop/Whale_Kaggle/Detection/keras-retinanet-master/build/lib.linux-x86_64-2.7/test_cropped/*')):
        start = time.time()
        image = cv2.imread(filename, 0)
        image = preprocess(image)
        image = image.astype('float32').reshape([1, IMG_HEIGHT, IMG_WIDTH, 1]) / 255.

        encoded_test = np.array(encoder_model.predict(image)[0])

        dist = (encoded_training - encoded_test) ** 2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)

        min_dist_indices = dist.argsort()[:5]

        y_predicted = y_training[min_dist_indices]
        list_names = [integer_to_name_map[x] for x in y_predicted]

        f = open('submission.csv', 'a')
        f.write('{},{} {} {} {} {}\n'.format(filename.split('/')[-1], list_names[0], list_names[1], list_names[2], list_names[3], list_names[4]))

        ending = time.time()
        print(i, 'Processing Time: ', ending-start)



    # f = open('submission.csv', 'w')
    # f.write('ImageID,label\n')
    #
    #
    # for i in tqdm(range(labels.shape[1])):
    #     l = labels[:, i]
    #     # voting between different models predictions
    #
    #
    #
    # f.write('{},{}\n'.format(i + 1, mode(l)[0][0]))
