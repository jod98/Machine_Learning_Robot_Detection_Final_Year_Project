# Author: Jordan O Donnell
# Date: 08/04/2020
# Description: This program peforms object detection on a Tensorflow-train neural network (in this case variations of SSD-MobileNet models were used)
#              It loads the classifier and uses it to perform object detection on an image.
#              It draws boxes, scores, and labels around the objects of interest in the image.
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
import glob
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
    
# Model directory path. Uncomment line to test using different model
MODEL_NAME = 'TFLite_model/inference_graph/ssd_mobilenet_v2_quantized_300x300_coco'
#MODEL_NAME = 'TFLite_model/inference_graph/ssd_mobilenet_v2_quantized_160x160_coco'
#MODEL_NAME = 'TFLite_model/inference_graph/ssd_mobilenet_v2_quantized_80x80_coco'
#MODEL_NAME = 'TFLite_model/inference_graph/ssdlite_mobilenet_v2_300x300_coco'
#MODEL_NAME = 'TFLite_model/inference_graph/ssdlite_mobilenet_v2_160x160_coco'
#MODEL_NAME = 'TFLite_model/inference_graph/ssdlite_mobilenet_v2_80x80_coco'

CURRENT_PATH = os.getcwd() # Current direectory path
PATH_TO_FROZEN_INFERENCE = os.path.join(CURRENT_PATH,MODEL_NAME,'detect.tflite') # Path to frozen detection graph (TFLite version) - model (includes 1. Graph definition 2 . Trained paramaters)
PATH_TO_LABELS = os.path.join(CURRENT_PATH,MODEL_NAME,'labelmap.txt') # Path to label map file

IMAGE_DIR = 'TEST_IMAGES_OWN_MODEL_FULL_TEST' # Image directory path

#Path to image
PATH_TO_IMAGES = os.path.join(CURRENT_PATH,IMAGE_DIR)
images = glob.glob(PATH_TO_IMAGES + '/*')

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

# Iterating through all images in the test-set while calculating inference time
for image_path in images:
    
    # Load image, Define RGB for each pixel, Expand image to set dimensions/shape
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

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

    # Draw bounding box around object
    for i in range(len(scores)):
        if ((scores[i] > float(0.5)) and (scores[i] <= 1.0)):

            # Get bounding box coordinates (Interpreter can return coordinates outside dimensions, need to force them to be within image using max() and min())
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            # Draw bounding box around object
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # To display image with detections uncomment the rest of the commented lines
    
    # All the results have been drawn on the image, now display the image
    cv2.imshow('SSD_MobileNet_V2_Quantized_300x300_COCO on Test Set', image)

    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Computing average FPS and printing to terminal
averageFPS=totalFPS/num_Frames
print('Average FPS: ' + str(averageFPS))

# Clean up
cv2.destroyAllWindows()
