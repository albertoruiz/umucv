#!/usr/bin/env python

# https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/python/object_detector.ipynb


import cv2          as cv
from umucv.stream import autoStream
from umucv.util import check_and_download
import math
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

check_and_download("efficientdet.tflite","https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite")


# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)


# STEP 5: Process the detection result. In this case, visualize it.

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv.putText(image, result_text, text_location, cv.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image


for key,frame in autoStream():
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    detection_result = detector.detect(mpimage)
    
    image_copy = np.copy(mpimage.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB)
    
    cv.imshow("object detection", rgb_annotated_image)

