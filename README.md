# person-detection-faster-rcnn
Person and object detection using pretrained Faster R-CNN with a ResNet-50 + FPN backbone in PyTorch.

Person Detection with Faster R-CNN

A simple computer vision project for detecting people and other objects in images using a pretrained Faster R-CNN model with a ResNet-50 + FPN backbone.

Overview

This project demonstrates object detection using PyTorch and Torchvision. A pretrained Faster R-CNN model is loaded and used to analyze an input image.

The model:

Takes an input image

Preprocesses the image using Torchvision transforms

Detects objects using Faster R-CNN

Produces bounding boxes, class labels, and confidence scores

Filters predictions using a confidence threshold of 0.50

Draws bounding boxes and labels on the detected objects

Note: This project uses a pretrained model for inference. The model is not trained from scratch in this notebook.

Model Architecture

The project uses:

Faster R-CNN + ResNet-50 + FPN

Faster R-CNN — object detection framework

ResNet-50 — extracts visual features from the image

FPN (Feature Pyramid Network) — helps detect objects at different scales

COCO dataset classes — provides the object categories used by the pretrained model

How It Works

Input Image
     ↓
Image Preprocessing
     ↓
Faster R-CNN
(ResNet-50 + FPN)
     ↓
Object Predictions
     ↓
Bounding Boxes + Labels + Scores
     ↓
Confidence Filtering
     ↓
Final Detection Visualization

Main Steps

1. Install Libraries

pip install torch torchvision matplotlib pillow

2. Import Libraries

The project uses PyTorch, Torchvision, Matplotlib, PIL, and related libraries for model inference and visualization.

3. Load the Pretrained Model

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

model.eval()

4. Load COCO Classes

The model uses COCO categories to identify detected objects.

COCO_CLASSES = weights.meta['categories']

For example:

ID 1  → person
ID 18 → dog
ID 32 → tie

5. Preprocess the Image

transform = weights.transforms()
image_tensor = transform(image)

The image is converted into the format expected by the pretrained model.

6. Run Object Detection

with torch.no_grad():
    predictions = model([image_tensor])

The model returns predictions containing:

boxes — coordinates of detected objects

labels — class IDs

scores — confidence scores

7. Apply Confidence Threshold

Only predictions with confidence ≥ 0.50 are kept.

confidence_threshold = 0.50
keep = scores >= confidence_threshold

This removes low-confidence detections.

8. Visualize Results

The final result displays a bounding box around the detected object together with its class name and confidence score.

Example:

person: 1.00

Example Result

For the example image used in this project, the model detected:

Object: person
Confidence: 0.9996

After applying the 0.50 threshold, the high-confidence person detection remains.

Technologies Used

Python

PyTorch

Torchvision

Faster R-CNN

ResNet-50

Feature Pyramid Network (FPN)

COCO Dataset

Matplotlib

Pillow

Jupyter Notebook / Google Colab

Project Structure

Person-Detection-Faster-RCNN/
│
├── Person_detection.ipynb
├── new.png
├── README.md
└── visualization.png

How to Run

Open Person_detection.ipynb in Jupyter Notebook or Google Colab.

Install the required libraries.

Run the import and model-loading cells.

Upload an image.

Run the preprocessing and inference cells.

Apply the confidence threshold.

View the final image with the detected objects and bounding boxes.

Future Improvements

Detect multiple objects in different images

Use GPU acceleration for faster inference

Add real-time webcam detection

Test different confidence thresholds

Extend the project to video object detection

Fine-tune the model on a custom dataset

License

This project is intended for educational and learning purposes.
