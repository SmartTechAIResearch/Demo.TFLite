#include <Arduino.h>

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model.h"

// Create a memory pool for the nodes in the network
constexpr int tensor_pool_size = 20 * 1024;
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

float temp_array[] = {0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.04,0.73,1.00,0.99,0.99,0.90,0.52,0.52,0.12,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.04,0.28,0.95,0.99,0.99,0.89,0.91,0.99,0.99,0.99,0.65,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.11,0.65,0.99,0.99,0.92,0.36,0.00,0.05,0.56,0.99,0.99,0.59,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.80,0.99,0.92,0.60,0.17,0.00,0.00,0.19,0.88,0.99,0.71,0.06,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.64,0.99,0.91,0.24,0.00,0.00,0.00,0.02,0.70,0.99,0.99,0.24,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.19,0.99,0.99,0.30,0.00,0.00,0.00,0.17,0.78,0.99,0.99,0.99,0.24,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.35,0.99,0.89,0.13,0.00,0.00,0.39,0.91,0.96,0.86,0.99,0.80,0.05,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.66,0.99,0.81,0.38,0.38,0.81,0.92,0.95,0.13,0.62,0.99,0.57,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.24,0.99,0.99,0.99,0.99,0.99,0.78,0.09,0.04,0.78,0.91,0.16,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.10,0.51,0.88,0.99,0.99,0.56,0.04,0.00,0.32,0.99,0.80,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.99,0.99,0.55,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.32,0.99,0.86,0.14,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.82,0.99,0.38,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.09,0.99,0.97,0.31,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.47,0.99,0.78,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.13,0.85,0.96,0.10,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.81,0.99,0.95,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.13,0.78,0.98,0.99,0.36,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.99,0.99,0.56,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.99,0.78,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                      0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00};

void setup() {
  Serial.begin(9600);

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
  Serial.print("  Number of dimensions: ");
  Serial.println(input->dims->size);
  Serial.print("  Dim 1 size: ");
  Serial.println(input->dims->data[0]);
  Serial.print("  Dim 2 size: ");
  Serial.println(input->dims->data[1]);
  Serial.print("  Input type: ");
  Serial.println(input->type);


  Serial.println("Output");
  Serial.print("  Number of dimensions: ");
  Serial.println(output->dims->size);
  Serial.print("  Dim 1 size: ");
  Serial.println(output->dims->data[0]);
  Serial.print("  Dim 2 size: ");
  Serial.println(output->dims->data[1]);
  Serial.print("  Output type: ");
  Serial.println(output->type);
}
void loop() {
  input->data.f = temp_array;
 
  Serial.println("Processing data");
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  output = interpreter->output(0);

  for (int i = 0; i < 10; i++) {
    Serial.print("[");
    Serial.print(i);
    Serial.print("]: ");
    Serial.println(output->data.f[i],2);
  }
}
