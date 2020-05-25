# ESP32 + MNIST Digits with arduinoIDE
Here you will find the documentation to replicate this project using the ArduinoIDE. 

## Setting up the ArduinoIDE for succes
First of all, after cloning the repository, you will have to move the folder ['tensorflow_lite'](https://github.com/sizingservers/Demo.TFLite/tree/master/arduinoide/tensorflow_lite) to the arduino library folder.
Next up, you should add the capability to use the ESP32, in order to do this you will have to add the following link to the 'Additional Boards Manager':
`https://dl.espressif.com/dl/package_esp32_index.json`
This can be done File -> Preferences -> AdditionalBoards Manager URLs
![Boards manager](https://i.imgur.com/KSvrlPE.png)

Now you should be able to find the ESP32 in the list of available boards, Tools -> Board:... -> ESP DEV Module
![](https://i.imgur.com/m7tuNUe.png)

Make sure that all the settings under the tab 'tools' are equal to those in the screenshot.
![](https://i.imgur.com/G3Tp85O.png)

## The code
The project contains a few files with the complementary header files, these filles are all comming togheter in the "ArduinoCode". I will be going over every single file explaining what it does and why it's needed.

### model.* file
Obviously, in the model files, you can find everything related to the tflite model. These files aren't realy difficult to understand, they contain a C-array that represents the model. On arduino you do have to specify some the allocation of the model, this is done by adding a Data alignment attribute. this attribute will specify how and where the model should be located in memory.

### ArduinoCode.ino
This is the hearth of the project, here everything comes togheter. First of all, we have all the imports that need to be done to get all the tensorflow methods, model files and other...
``` Arduino
#include <TensorFlowLite.h>
#include "arduino.h"

#include "model.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

In order to use all the Tensorflow operations, I had to create a namespace with preallocated variables. These variables will be used throughout the code and are handeling everything involving the predicting and error reporting process. 
``` Arduino
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
} 
```
#### Void setup()
The next step is configuring all the variables in the Void setup().
This code will try to connect to the sensor using I2C, in case the sensor isn't available or not connected properly it will also print the error. Next, I set up the logging and try to load the model:
``` Arduion
static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_mnist);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
```
After that, I will prepaire the operations resolver to be able to handle all the available ops. It is possible to only select specific ops, however I didn't feel the need to do so since there is no memory restriction.
``` Arduino
static tflite::ops::micro::AllOpsResolver resolver;
```
As last step in the setup, I build the interpreter and print more information about the In- and Output. With this information you can quickly verify if it all loaded propperly.
``` Arduino
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Input");
  Serial.print("Number of dimensions: ");
  Serial.println(input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(input->type);


  Serial.println("Output");
  Serial.print("Number of dimensions: ");
  Serial.println(output->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(output->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(output->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(output->type);
```
#### Void loop()
As some say, this is where the magic happens, however it's just some code. The following code will only be executed when the serial port recieves an input. It will save the byte array it recieves, once the complete array is recieved continue to input the data so the interpreter can invoke and return the prediction.
``` Arduino
  if (Serial.available() > 0) {
    size_t data_length = Serial.readBytes(char_array, char_array_size);
    Serial.printf("Data length: %u \n", data_length);

    input->data.raw = char_array;
  
    Serial.println("Processing data");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed");
      return;
    }

    output = interpreter->output(0);

    for (int i = 0; i < 10; i++) {
      Serial.printf("[%d]: %f \n", i, output->data.f[i]);
    }
  }

```

### The python demo code
This code is created to interact with the ESP32, the ESP is set up to receive a bytearray that contains a 28x28 represenatation of a created digit. In order to implement the possibility to make the digit, we used pygame. With pygame you can create various projects, however sortoff created paint.

The code conosists out of a few parts, first of all the imports: 
``` python
import serial
from threading import Thread
import time
import numpy as np
import pygame 
import cv2
import struct
```
 In order to be able to continiously recieve data from the esp, we used threading. In the thread we made a function that printed the ESP output. 
 ``` python
def serial_output():
    while running:
        try:
            espData = ser.readline().decode('utf-8')[:-2]
            if espData != "": print(espData, end=None)
        except:
            pass
```
In order to be able to send the digits to the ESP, we have to convert it to a bytearray, this is done using the following function. This function will retreave the drawing of a digit from the pygame screen.
 ``` python
def predict_digit():
    view = pygame.surfarray.array3d(screen)
    view = view.transpose([1, 0, 2])
    img = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    data = np.asarray(resized)
    data_list = list(data.flatten()/255)
    data_bytes = bytearray()

    for i in range(784):
        n = data_list[i]
        n = struct.pack('f', n)
        for b in n:
            data_bytes.append(b)
                    
     # print(f'Data: {data_bytes}')
    print(f'Length data sent: {len(data_bytes)}')
    ser.write(data_bytes)
    ser.flush()
```
The Pygame itself runs in a single while loop, it basicly awaits input events in order to do something. the following input events are implemented:
- MouseDown -> user is able to draw
- Q (or A) -> the progam is shut down
- C -> clears drawing board
- p -> predicts the drawn digit

```python
while True:
    e = pygame.event.wait()
    if e.type == pygame.QUIT:
        raise StopIteration
    if e.type == pygame.MOUSEBUTTONDOWN:
        pygame.draw.circle(screen, color, e.pos, radius)
        draw_on = True
    if e.type == pygame.MOUSEBUTTONUP:
        draw_on = False
    if e.type == pygame.KEYDOWN:
        if e.key == pygame.K_p:
            #pygame.image.save(screen,digitfile)
            predict_digit()
        elif e.key == pygame.K_c:
            screen.fill((0,0,0))
            pygame.display.update()
        elif e.key == pygame.K_q or e.key == pygame.K_a: #pygame doesn't understand qwerty/azerty ?
            raise StopIteration
    if e.type == pygame.MOUSEMOTION:
        if draw_on:
            pygame.draw.circle(screen, color, e.pos, radius)
            roundline(screen, color, e.pos, last_pos,  radius)
        last_pos = e.pos
    pygame.display.flip()
```





