import tensorflow as tf
import datetime
import os
import pandas as pd
import numpy as np
import sys

import argparse

parser = argparse.ArgumentParser(description='Demo code for the MNIST TFLite workshop.')

parser.add_argument('--data', type=str, help="The data you want to use, either 'mnist' or 'emnist'", dest='data', default='mnist')

args = parser.parse_args()


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    import gzip
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

# load and preprocess data

# dataset = 'emnist' # or 'emnist'
dataset = args.data
output_layers = 10 # default, will be overridden

IMAGE_SHAPE = (28, 28, 1)


if dataset == 'mnist':
    output_layers = 10
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    ## Alias
    y_train_values = y_train
    y_test_values = y_test

    ## Normalize and reshape

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0],28,28,1)
    x_test = x_test.reshape(x_test.shape[0],28,28,1)

elif dataset == 'emnist':

    ## Character mapping to know the labels and the ASCII characters
    mapp = pd.read_csv("emnist/emnist-letters-mapping.txt", delimiter = ' ', \
                   index_col=0, header=None, squeeze=True)

    ## Don't take too much classes, because our model will be too large.
    
    emnist_distinct_upper = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T'] # Characters with distinct uppercase
    emnist_nondistinct_upper = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z'] # Characters with non-distinct uppercase, too large to use in this case
    emnist_nondistinct_upper_partial = ['c', 'i', 'k', 'l', 'm', 'o', 'p', 's', 'v', 'x', 'z'] # A selection of 11 characters to use as our model can not be too big

                             # always uppercase  # Choose one of ('emnist_distinct_upper', 'emnist_nondistinct_upper_partial')
    emnist_characters = [ord(i.upper()) for i in emnist_distinct_upper]

                                        # COLUMN 1 is uppercase, 2 is lowercase
    emnist_characters_ascii = list(mapp[mapp[1].isin(emnist_characters)].index)

    output_layers = len(emnist_characters_ascii) ## Get the amount of characters, because that will be our output layer.
    x_train, y_train = load_mnist('emnist', kind='emnist-letters-train') # X stands for features, y for targets
    x_test, y_test = load_mnist('emnist', kind='emnist-letters-test')
    
    ## Only use the characters that we need, so mask them
    
    train_mask = np.isin(y_train, emnist_characters_ascii)
    x_train = x_train[train_mask]
    y_train = y_train[train_mask] - 1

    test_mask = np.isin(y_test, emnist_characters_ascii)
    x_test = x_test[test_mask]
    y_test = y_test[test_mask] - 1
    

    # Because we are selecting values that can be higher than 11, we can just normalize them to a value between 0 and 10.

    y_train_labels, y_train_values = np.unique(y_train, return_inverse=True)
    y_test_labels, y_test_values = np.unique(y_test, return_inverse=True)

    LABELS = [chr(mapp.iloc[i][1]) for i in y_train_labels]
    print("Make sure to copy these labels to the Output classification program. (Arduino or test-python)")
    print(LABELS)


    ## Normalize, rotate and reshape
    def rotate(image):
        image = image.reshape(*IMAGE_SHAPE)
        image = np.fliplr(image)
        image = np.rot90(image)
        return image

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.asarray(x_train)
    x_train = np.apply_along_axis(rotate, 1, x_train)

    x_test = np.asarray(x_test)
    x_test = np.apply_along_axis(rotate, 1, x_test)
else:
    print("You have chosen an invalid dataset.")
    print("Exitting now")
    sys.exit(0)



print ("x_train:", x_train.shape)
print ("x_test:", x_test.shape)




# Save the resulted model
# model.save(dataset + "-model.h5")

model = tf.keras.models.load_model(dataset + "-model.h5")

#-------------------------------------------------------------------
# or quantize aware training to directly convert it to a tflite model
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# 'quantize_model' requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

q_aware_model.fit(x_train, y_train_values,
                  batch_size=500, epochs=1, validation_split=0.1)

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()
open(dataset + "-quantized_model.tflite", "wb").write(quantized_tflite_model)