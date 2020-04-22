# Machine_Learning_Robot_Detection_Final_Year_Project
RoboEireann is a NUI (National University of Ireland) Maynooth funded research project that features a soccer team composed of programmable autonomous Nao robots. The team annually competes in the RoboCup competition against international competitors. This project investigates RoboEireann’s current robot detection program which is a program that aids these devices to detect other robots in their environment. Current issues relate to its inability to produce high accuracy results. The aim is to improve the current program by utilising machine learning principles. An enhanced version of the RoboEireann’s SPL (Standard Platform League) dataset was created and provided as input to an object detection neural network which compared many pre-trained models. Experimentation of various machine learning frameworks, architectures and object detection algorithms was conducted to determine the optimal solution. This solution will be integrated into each robot in real-time during a competitive match. These findings have significant implications for the development of robotics and artificial intelligence.

## Repository Details
This repository provides the user with the ability to train and evaluate their own neural network to perform the application of their choice. In my case I created a convolutional neural network that integrates supervised learning. My dataset consists of 4415 training images and 491 testing images in which aid to train the network alongside the corresponding labels (examples of these images are included in the 'Sample_Images' folder). My networks were trained on both TensorFlow and TensorFlow Lite (lightweight solution) on two different platforms a desktop PC with an integrated GPU and a Raspberry Pi 3 to provide a comparison. 

The 'Results' and 'Demonstration' folders highlight the outcome of this project.

### Credits
1. Google's Open-Sourcec Framework, TensorFlow:
- https://github.com/tensorflow/models
- https://github.com/tensorflow/tensorflow

2. Edje Electronics (Provides very intuitive tutorials on the setup process)
- https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
- https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
- https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md
