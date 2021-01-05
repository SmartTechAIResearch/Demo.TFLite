// Undefine predefined Arduino max/min defs so they don't conflict with std methods
#if defined(min)
#undef min
#endif

#if defined(max)
#undef max
#endif


#include <TensorFlowLite.h>
#include "arduino.h"

#include "model.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

const size_t char_array_size = 3136;

char char_array[char_array_size];

void setup() {
  Serial.println("Beginning setup");
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
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

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::ops::micro::AllOpsResolver resolver;

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
  Serial.flush();
  Serial.end();
  Serial.begin(250000);
  
  Serial.print("Number of Input dimensions: ");
  Serial.println(input->dims->size);
  Serial.printf("Image size: (%d, %d) \n", input->dims->data[1], input->dims->data[2]);

  Serial.print("Number of Output dimensions: ");
  Serial.println(output->dims->size);
  Serial.printf("Output size: (%d, %d) \n", output->dims->data[0], output->dims->data[1]);
}

void loop() {
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

    // const char *labels[] = {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
    const char *labels[] = {"A", "B", "D", "E", "F", "G", "H", "N", "Q", "R", "T"};

    for (int i = 0; i < output->dims->data[1]; i++) {
      Serial.printf("[%s]: %f \n", labels[i], output->data.f[i]);
    }
  }
}
