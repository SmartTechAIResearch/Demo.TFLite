# (Re-)Training the MNIST Model

## Data choice
We have two choices of data, we can use the **MNIST** data which is just for digit-recognition. Or we use the **E-MNIST** which also includes support for letters.

Whichever dataset you choose will not alter this demo, we will be doing both of them anyways.

## Training
To train the model, we can simply run the `trainAndConvertModel.py` which automatically converts the model to a `.tflite` model as well.
It recompiles the model, this time Quantized Aware, which ensures us that the model is a `.tflite` variant.

You can test the `.h5` model and it's accuracy using the `pygame-test-no-esp.py`. Which is used to test the model without using the `.tflite` model.
Note that the code is slightly different, because the predictions come from the Python API instead of the Arduino code.

Evaluate the model and fine-tune if necessary.

### Notes
- We need to augment the handwritten digits of the MNIST model, we will rotate and slightly shift horizontally and vertically so that they are more realistic towards the Pygame drawings.
- For the EMNIST characters, we do not have to do that, because the images are already quite robust.
- The EMNIST characters are both uppercase and lowercase, which is quite complex to train, the model should be a little bit complexer to be certain.
- Some letters have very distinct uppercase and lowercase characters, which is interesting to train. Examples of those very distinct are: [A / a, B / b, T / t]. Whereas others are more difficult to see the difference between lowercase and uppercase: [C \ c, S \ s, U \ u]
- The ESP32 does not have that much memory (16MB) which means that our model can not be too big. It is therefore not possible to include too much characters of the EMNIST dataset. We have limited to 11 characters with very distinct uppercase and lowercase. 

## Converting to C-Array
After the training, we should convert our `.tflite`-model to a C-Array.
We can do this by performing a Linux command `xxd -i model.tflite > model.cc`. The output you will see will look like this:
```cpp
unsigned char _tmp_rs_1609861507764872247[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  ...
  0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00, 0x00, 0x00,
};
unsigned int _tmp_rs_1609861507764872247_len = 16928;
```

We can copy all these bytes into the `model.cpp` inside the `ArduinoCode` directory. Search for the corresponding lines and copy the content.

### I don't have a Linux machine ready 😢😭
Perhaps you have one of the following options instead: [Docker](https://www.docker.com/) or [Ubuntu on Windows](https://ubuntu.com/tutorials/ubuntu-on-windows#1-overview).


#### Docker
If you can work with Docker, you can pull `nathansegers/xxd-web:latest` and run it: `docker run -d -p 8000:8000 nathansegers/xxd-web:latest`

Good thing: We have made this Docker image available on an Azure webapp - [xxd-converter.azurewebsites.net](xxd-converter.azurewebsites.net).

#### Ubuntu for Windows
If you have Ubuntu for Windows, you can use the command as described above.

