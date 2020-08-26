import tensorflow as tf
import datetime
import os
import numpy as np

#load and preprocess data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

#create model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(8, kernel_size=(3,3), strides=(1,1),
                         activation='relu',input_shape=(28,28,1)),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Softmax()
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()

#add datagen to make the model more accepting for the 'mouse' written digits
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=25,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

#Train the model using the augmentation datagenerator
history = model.fit_generator(
      datagen.flow(x_train,y_train, batch_size=64),
      steps_per_epoch=100,
      epochs=150,
      verbose=2)

#validate model to verify the result
model.evaluate(x_test,  y_test, verbose=2)

#Save the resulted model
model.save("model_augm.h5")

#-------------------------------------------------------------------
#or quantize aware training to directly convert it to a tflite model
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# 'quantize_model' requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

train_images_subset = x_train
train_labels_subset = y_train

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=1, validation_split=0.1)

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()
open("quantized_model.tflite", "wb").write(quantized_tflite_model)