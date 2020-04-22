# Author: Jordan O Donnell
# Date: 05/04/2020
# Description: This program peforms object detection on a Tensorflow-train neural network (in this case variations of SSD-MobileNet models were used)
#              It loads the classifier and uses it to perform object detection on an image.
#              It draws boxes, scores, and labels around the objects of interest in the image.
#
# Credits (I created this code with the help of the following resources):
#         - TensorFlow's open source GitHub repository (https://github.com/tensorflow/models)
#         - Evan Juras's (Edje Electronics) open source GitHub repository (https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

# Importing dependencies
import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import glob
import time
totalFPS=0
averageFPS=0
num_Frames=0
fullRenderTime = 0
FPS = 0

# Required as notebook stored in 'object_detection' folder
sys.path.append("..")

# Importing utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Initialising global variables.
fullRenderTime = 0
FPS = 0

# Model directory path. Uncomment line to test using different model
#MODEL_NAME = 'TF_model/inference_graph/ssd_mobilenet_v2_quantized_300x300_coco'
#MODEL_NAME = 'TF_model/inference_graph/ssd_mobilenet_v2_quantized_160x160_coco'
#MODEL_NAME = 'TF_model/inference_graph/ssd_mobilenet_v2_quantized_80x80_coco'
MODEL_NAME = 'TF_model/inference_graph/ssdlite_mobilenet_v2_300x300_coco'
#MODEL_NAME = 'TF_model/inference_graph/ssdlite_mobilenet_v2_160x160_coco'
#MODEL_NAME = 'TF_model/inference_graph/ssdlite_mobilenet_v2_80x80_coco'


#MODEL_NAME = 'TF_model/inference_graph/ssd_mobilenet_v1_quantized_300x300_coco'
#MODEL_NAME = 'TF_model/inference_graph/ssdlite_mobilenet_v2_coco'
#MODEL_NAME = 'TF_model/inference_graph/ssd_mobilenet_v2_coco'
#MODEL_NAME = 'TF_model/inference_graph/ssd_mobilenet_v1_coco'

IMAGE_DIR = 'TEST_IMAGES_OWN_MODEL_FULL_TEST' # Image directory path
CURRENT_PATH = os.getcwd() # Current directory path
PATH_TO_FROZEN_INFERENCE = os.path.join(CURRENT_PATH,MODEL_NAME,'frozen_inference_graph.pb') # Path to frozen detection graph - model (includes: 1. Graph definition 2. Trained parameters)
PATH_TO_LABELS = os.path.join(CURRENT_PATH,'training','labelmap.pbtxt') # Path to label map file

# Path to image
PATH_TO_IMAGES = os.path.join(CURRENT_PATH,IMAGE_DIR)
images = glob.glob(PATH_TO_IMAGES + '/*')

NUM_CLASSES = 4 # Number of classes to identify

# Load the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_INFERENCE, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
      
    sess = tf.Session(graph=detection_graph) # Create a session  

# Input tensor = image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors = detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') # xmin, ymin, xmax, ymax
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') # confidence scores
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') # 'Ball', 'Robot', 'Goal_Post', 'Penalty_Spot'
num_detections = detection_graph.get_tensor_by_name('num_detections:0') # Calculate no. of objects detected

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Iterating through all images in the test-set while calculating inference time
for image_path in images:

    # Load image, Define RGB for each pixel, Expand image to set dimensions/shape
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Performing detection
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    totalFPS = totalFPS + frame_rate_calc
    num_Frames = num_Frames+1

    # Drawing Detection Results
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.50)

    # To display image with detections uncomment the rest of the commented lines

    cv2.imshow('SSD_MobileNet_V2_Quantised_80x80_coco on Test Set', image)
    if cv2.waitKey(1) == ord('q'):
        break

averageFPS=totalFPS/num_Frames
print('Average FPS: ' + str(averageFPS))

cv2.destroyAllWindows()
