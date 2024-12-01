# YOLOv3_Object_Detection
This project implements object detection using YOLOv3 and OpenCV. It identifies 80 object categories from the COCO dataset in images, displaying bounding boxes, class labels, and confidence scores. The model uses pre-trained YOLOv3 weights and is fully customizable. Simply add the required files, run the script, and view the detected objects.
Description:
This project demonstrates real-time object detection using YOLOv3 and OpenCV. It leverages the pre-trained YOLOv3 model with the COCO dataset to identify objects in an image. The program displays bounding boxes, class labels, and confidence scores on the detected objects.

Features:

Detects 80 object categories from the COCO dataset.
Visualizes detected objects with bounding boxes and confidence scores.
Easily customizable for other models or datasets.
Dependencies:

Python 3.6+
NumPy
OpenCV
Matplotlib
Usage:

Place the required files (yolov3.weights, yolov3.cfg, coco.names) in the project directory.
Add the input image (e.g., IMG_4934.JPG).
Run the script:
python yolo_object_detection.py
View the output image with detected objects.
