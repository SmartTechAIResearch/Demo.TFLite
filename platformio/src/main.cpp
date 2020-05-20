#include <Arduino.h>

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model.h"

// Create a memory pool for the nodes in the network
constexpr int tensor_pool_size = 16 * 1024;
uint8_t tensor_pool[tensor_pool_size];

// Define the model to be used
const tflite::Model* model;

// Define the interpreter
tflite::MicroInterpreter* interpreter;

// Define the error reporter
tflite::ErrorReporter* error_reporter;

// Input/Output nodes for the network
TfLiteTensor* input;
TfLiteTensor* output;

union u_float {
  byte b[4];
  float fval;
} u;

const size_t float_array_size = 784;
float float_array[float_array_size];

const size_t byte_array_size = 3136;
byte byte_array[byte_array_size];

void setup() {
  Serial.begin(250000);

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay. 
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_mnist);

  // This pulls in all the operation implementations we need.
  static tflite::ops::micro::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_pool, tensor_pool_size, error_reporter);
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
  Serial.printf("  Number of dimensions: %d \n", input->dims->size);
  Serial.printf("  Dim 1 size: %d \n", input->dims->data[0]);
  Serial.printf("  Dim 2 size: %d \n", input->dims->data[1]);
  Serial.printf("  Input type: %d \n", input->type);


  Serial.println("Output");
  Serial.printf("  Number of dimensions: %d \n", output->dims->size);
  Serial.printf("  Dim 1 size: %d \n", output->dims->data[0]);
  Serial.printf("  Dim 2 size: %d \n", output->dims->data[1]);
  Serial.printf("  Output type: %d \n", output->type);
}
void loop() {
  if (Serial.available() > 0) {
    size_t data_length = Serial.readBytes(byte_array, byte_array_size);

    Serial.printf("Data length: %u \n", data_length);
    
    for (int i = 0; i < byte_array_size; i += 4) {
      for (int f = 0; f < 4; f++) {
        u.b[f] = byte_array[i + f];
      }
      float data_float = u.fval;
      float_array[i / 4] = data_float;
    }

    input->data.f = float_array;
  
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
}
