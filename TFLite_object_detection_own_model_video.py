# Author: Jordan O Donnell
# Date: 08/04/2020
# Description: This program peforms object detection on a Tensorflow-train neural network (in this case variations of SSD-MobileNet models were used)
#              It loads the classifier and uses it to perform object detection on a video.
#              It draws boxes, scores, and labels around the objects of interest in the video.
#
# Credits (I created this code with the help of the following resources):
#         - TensorFlow's open source GitHub repository (https://github.com/tensorflow/tensorflow)
#         - Evan Juras's (Edje Electronics) open source GitHub repository (https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import time

# Initialising global variables.
totalFPS=0
averageFPS=0
num_Frames=0
imW, imH = int(1280), int(720) # Specifying resolution of frame

# Importing TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    #from tensorflow.lite.python.interpreter import Interpreter     

MODEL_NAME = 'TFLite_model/inference_graph/ssd_mobilenet_v2_quantized_300x300_coco' # Model directory path

CURRENT_PATH = os.getcwd() # Current direectory path
PATH_TO_FROZEN_INFERENCE = os.path.join(CURRENT_PATH,MODEL_NAME,'detect.tflite') # Path to frozen detection graph (TFLite version) - model (includes 1. Graph definition 2 . Trained paramaters)
PATH_TO_LABELS = os.path.join(CURRENT_PATH,MODEL_NAME,'labelmap.txt') # Path to label map file

VIDEO_PATH = 'TEST_VIDEOS_OWN_MODEL/Robocup_2019_SPL_Final-HTWKvsB-Human_1280x720_CLIP.mp4'# Video directory path

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Interpreter allows us to perform inference on a TFLite model
interpreter = Interpreter(model_path=PATH_TO_FROZEN_INFERENCE)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#If using a non quantised model
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Open video file, Get width, Get height
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

while(video.isOpened()):
    
    # Load frame, Define RGB for each pixel, Expand frame to set dimensions/shape
    ret, frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    # Performing detection
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    totalFPS = totalFPS + frame_rate_calc
    num_Frames = num_Frames+1
    

    # Drawing Detection Results
    for i in range(len(scores)):
        if ((scores[i] > float(0.5)) and (scores[i] <= 1.0)):

            # Get bounding box coordinates (Interpreter can return coordinates outside dimensions, need to force them to be within image using max() and min())
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            # Draw bounding box around object
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object Detection - Video', cv2.resize(frame, (1280,720)))
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Computing average FPS and printing to terminal
averageFPS=totalFPS/num_Frames
print('Average FPS: ' + str(averageFPS))

# Clean up
video.release()
cv2.destroyAllWindows()