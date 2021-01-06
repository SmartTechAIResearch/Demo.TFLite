# Demo.TFLite

Demo and workshop to demonstrate the development flow for TFLite models.

## Requirements
- Arduino IDE is preferred.
- PlatformIO is also possible, but hasn't been updated.
- Works on Windows and Linux
- See Python requirements in the other directories.

## Content
The content is seperated in 3 main folders.
- `arduinoide` contains code to work with Tensorflow Lite through the Arduino IDE.
- `MNIST` contains the Python code to train and convert models using Tensorflow (Keras) and Tensorflow Lite.
- `platformIO` contains code specific for PlatformIO, which would allow to upload Arduino code through Visual Studio Code.

## Converting TFLite models to C-Arrays
Read more on this in the [`MNIST > README`](https://github.com/sizingservers/Demo.TFLite/tree/master/MNIST/Readme.md) file on the different ways to convert TFLite models to C-Arrays.
In short: Use either Linux, or a Docker container.
