import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

mapp = pd.read_csv("emnist/emnist-letters-mapping.txt", delimiter = ' ', \
                   index_col=0, header=None, squeeze=True)

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

#Load tflite model and initialize interpreter
dataset = 'emnist' # or 'emnist'
output_layers = 10 # default, will be overridden

if dataset == 'mnist':
  output_layers = 10
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif dataset == 'emnist':
  output_layers = 10
  x_test, y_test = load_mnist('emnist', kind='emnist-letters-test')
  x_test = x_test[y_test <= output_layers]
  y_test = y_test[y_test <= output_layers] - 1




x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0],28,28,1)

print(x_test.shape)

interpreter = tf.lite.Interpreter(model_path=dataset + "-quantized_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors. (can be used to debug any problems)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(interpreter.get_tensor_details())

#Example on how you can use this to debug
# print(output_details[0])

#This will output something like this:
#{'name': 'Identity', 'index': 10, 'shape': array([ 1, 10]), 'shape_signature': array([ 1, 10]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}


# # Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# #invoking the interpreterd, followed by printing the predicted result
interpreter.invoke()
# print(interpreter.get_tensor(output_details[0]['index']))

#predict every row from x_test
lst_results = []
for row in x_test:
    x = tf.expand_dims(np.array(row, dtype=np.float32),0)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    lst_results.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
    
#Wrap up by calculating the accuracy
correct = 0
for i in range(len(y_test)):
    if y_test[i] == lst_results[i]:
        correct += 1
        
print(f"Correct: {correct}") 
print(f"Total smaples: {len(y_test)}") 
print(f"Accuracy: {correct/len(y_test)}")


for i in range(100, 109):
    plt.subplot(330 + (i+1))
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    # plt.title(chr(mapp[y_test[i] + 1]))