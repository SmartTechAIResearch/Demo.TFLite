import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
tf.executing_eagerly()

#Load dataset and do minor preprocessing to recreate training data.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.float32(x_test.reshape(x_test.shape[0],28,28,1))
x_test = x_test / 255

#load previously saved tensorflow model (example for .h5 files)
model = load_model("model_augm.h5")

#method thats handles the representative data generation
def representative_dataset_gen():
  for x in x_test:
    x = tf.expand_dims(x,0)
    x = np.float32(x)
    yield [x]
    
#add model to the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#declare chosen optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

#enable represenative data gen (refers to the method above)
converter.representative_dataset = representative_dataset_gen

#Exectue the converter
quantized_model = converter.convert()

#save quantized model
open("quantized_model.tflite", "wb").write(quantized_model)

