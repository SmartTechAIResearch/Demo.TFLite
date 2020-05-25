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
