import numpy as np
import tensorflow as tf

#Load tflite model and initialize interpreter
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors. (can be used to debug any problems)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(interpreter.get_tensor_details())

#Example on how you can use this to debug
print(output_details[0])

#This will output something like this:
#{'name': 'Identity', 'index': 10, 'shape': array([ 1, 10]), 'shape_signature': array([ 1, 10]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}


# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

#invoking the interpreterd, followed by printing the predicted result
interpreter.invoke()
print(interpreter.get_tensor(output_details[0]['index']))


#(optional from this point on) test the accuracy of the tflite model on the test set
#loading the test set
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.float32(x_test.reshape(x_test.shape[0],28,28,1))
x_test = x_test/255

#predict every row from x_test
lst_results = []
for row in x_test:
    x = tf.expand_dims(row,0)
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
