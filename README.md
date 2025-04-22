# Object_Detection

In this project, we developed a custom object detection model using a **ResNet-50 backbone** and compared its performance against pre-trained YOLOv5 variants: **YOLOv5n (nano)**, **YOLOv5s (small)**, **YOLOv5m (medium)**, and **YOLOv5l (large)**. 

The evaluation was conducted on the **COCO dataset** using key performance metrics:
- **Recall**
- **Precision**
- **Intersection over Union (IoU)**

These metrics provide a comprehensive understanding of the models' accuracy, detection capabilities, and bounding box quality. The comparison highlights the trade-offs between the custom model and YOLOv5 variants in terms of accuracy and computational efficiency, offering insights into their suitability for real-world applications.

---

## Custom Model Architecture

The custom object detection model uses a **ResNet-50 backbone** to extract robust image features. It is tailored for object detection tasks on the COCO dataset and is designed to predict:
- **Fixed number of bounding boxes per image**
- **Class probabilities** for objects
- **Bounding box coordinates**

This model leverages:
- **CrossEntropy Loss** for classification
- **Smooth L1 Loss** for bounding box regression

While not as optimized as YOLO models, this modular design serves as an effective baseline for understanding the fundamentals of object detection.

---

### 1. Backbone: ResNet-50
- **Purpose:** 
  - Extracts high-level spatial features from input images.
- **Implementation:**
  - Pre-trained **ResNet-50** from `torchvision`, with fully connected layers removed:
    ```python
    self.backbone = torchvision.models.resnet50(weights="DEFAULT")
    self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
    ```
  - Outputs feature maps of size `[batch_size, 2048, H/32, W/32]`.
- **Key Properties:** 
  - Uses ImageNet-pretrained weights for robust feature extraction.

---

### 2. Classification Head
- **Purpose:** Predicts class probabilities for each bounding box.
- **Structure:**
  - Input: Flattened feature maps of size `[batch_size, 2048 * 7 * 7]`.
  - Layers:
    - Fully connected layer to reduce dimensionality.
    - **ReLU** for non-linearity and **Dropout** for regularization.
    - Final fully connected layer outputs logits shaped as `[batch_size, num_boxes_per_image, num_classes]`.
  - **Loss Function:** 
    - **CrossEntropy Loss** compares predicted class probabilities with ground truth labels.

---

### 3. Bounding Box Regression Head
- **Purpose:** Predicts bounding box coordinates for each detected object.
- **Structure:**
  - Input: Flattened feature maps of size `[batch_size, 2048 * 7 * 7]`.
  - Layers:
    - Fully connected layer to reduce dimensionality.
    - **ReLU** for non-linearity and **Dropout** for regularization.
    - Final fully connected layer outputs 4 values (`[x_min, y_min, x_max, y_max]`) for each bounding box.
  - **Loss Function:** 
    - **Smooth L1 Loss** measures the difference between predicted and ground truth coordinates.

---

### Loss Functions

In this model, two loss functions are combined to optimize predictions during training:
1. **Classification Loss:** Ensures correct class predictions for each bounding box.
2. **Regression Loss:** Ensures accurate bounding box predictions.

The total loss is computed as:

$$
\text{Total Loss} = \text{Classification Loss}+\text{Regression Loss}
$$

---

## Metrics

### Intersection over Union (IoU)
- **Definition:**
  - IoU quantifies the overlap between the predicted and ground truth bounding boxes.

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

- **Purpose:**
  - Evaluates the quality of predicted bounding boxes.
  - Higher IoU indicates better alignment with ground truth.

---

### Precision and Recall

| **Metric**   | **Focus**                   | **High Value Indicates**             |
|--------------|-----------------------------|--------------------------------------|
| **Precision** | Quality of predictions      | Most predictions are correct         |
| **Recall**    | Completeness of detections  | Most ground truth objects are detected |

#### **Precision**
Precision measures the proportion of correct predictions (True Positives) among all predictions made by the model:

$$
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$

#### **Recall**
Recall measures the proportion of actual objects (ground truth) that the model correctly detected:

$$
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
$$

---

## YOLO Model

**YOLO (You Only Look Once)** is a state-of-the-art object detection model that predicts bounding boxes and class probabilities in a single forward pass, making it highly efficient for real-time applications.

### Features:
- **Single-stage detector**: Combines detection and classification in one pass.
- **Grid-based detection**: Divides the image into a grid and assigns each cell responsibility for detecting objects.
- **Pretrained Weights**: YOLO models are pretrained on the **COCO dataset**, offering out-of-the-box detection for 80 object classes.

---

### YOLOv5 Pretrained Models

YOLOv5 offers multiple pretrained models optimized for various use cases:

| **Model Variant** | **Description**                                    | **Performance**                   |
|--------------------|----------------------------------------------------|------------------------------------|
| **YOLOv5n**        | Nano version, optimized for low-resource devices. | Smallest and fastest.             |
| **YOLOv5s**        | Small version, balances speed and accuracy.       | Ideal for lightweight tasks.      |
| **YOLOv5m**        | Medium version, more accurate but slower.         | Suitable for balanced use cases.  |
| **YOLOv5l**        | Large version, high accuracy but resource-intensive. | For high-performance applications. |

--- 🚀

##### Note: Only Yolov5s Model's results have been presented below but results for different yolo versions can be checked by running the "YoloV5__VF" notebook by changing the model name.

#### Average Precision and Recall at Different IoU Ranges

This below tables shows the evaluation metrics for object detection performance based on the Average Precision (AP) and Average Recall (AR) at different Intersection over Union (IoU) thresholds and for different object sizes. The metrics are computed for various configurations.

##### Average Precision (AP)

- **AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]** = 0.404
- **AP @[ IoU=0.50 | area=all | maxDets=100 ]** = 0.508
- **AP @[ IoU=0.75 | area=all | maxDets=100 ]** = 0.457
- **AP @[ IoU=0.50:0.95 | area=small | maxDets=100 ]** = 0.199
- **AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]** = 0.567
- **AP @[ IoU=0.50:0.95 | area=large | maxDets=100 ]** = 0.638

##### Average Recall (AR)

- **AR @[ IoU=0.50:0.95 | area=all | maxDets=1 ]** = 0.334
- **AR @[ IoU=0.50:0.95 | area=all | maxDets=10 ]** = 0.434
- **AR @[ IoU=0.50:0.95 | area=all | maxDets=100 ]** = 0.434
- **AR @[ IoU=0.50:0.95 | area=small | maxDets=100 ]** = 0.205
- **AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]** = 0.571
- **AR @[ IoU=0.50:0.95 | area=large | maxDets=100 ]** = 0.691

##### Summary

- The model demonstrates good overall performance, with particularly strong recall for larger objects.
- The **AP** is highest for large objects (0.638), and lowest for small objects (0.199).
- **AR** shows a consistent improvement with increased object size, peaking for large objects (0.691).


### Class Prediction vs. Ground Truth Table

| **Image ID** | **Predicted Class** | **Ground Truth Class** |
|--------------|---------------------|------------------------|
| 85329        | person              | person                 |
| 85329        | tie                 | tie                    |
| 297343       | stop sign           | stop sign              |
| 41888        | bird                | bird                   |
| 41888        | bird                | bird                   |
| 41888        | bird                | bird                   |
| 555705       | cat                 | cat                    |
| 555705       | cat                 | cat                    |

## Comparison between Yolov5 and Custom Model
### Yolov5
The below Image shows the Average Precision plot for different IoU and different sizes.
![AP](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/AP.png)

The below Image shows the Average Recall plot for IoU and different sizes.
![AR](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/AR.png)

The predicted and actual bouding boxes along with class labels are displayed below for two test images for yolov5s
1. A person wearing a Tie:
![yolo_p](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/yolo_p.png)
2. Stop Sign
![yolo_s](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/yolo_s.png)

### Custom Model
#### Training and Validation loss:
 - Training loss refers to the error or difference between the model’s predictions and the actual labels on the training dataset after each iteration or epoch of training. It gives us an indication of how well the model is fitting the training data.
 - Validation loss is the error or difference between the model’s predictions and the actual labels, but on a separate validation dataset that is not seen by the model during training. This dataset is used to evaluate the model's generalization performance.Validation loss is used to check if the model is overfitting or underfitting. Overfitting happens when the model performs very well on the training data but poorly on unseen validation data.

The graph below shows that both Training and validation losses decrease at start but after 3 or 4 epochs, validation loss doesn't decrease which means that our model is having some problem in generalizing for unseen data. As this is a custom model and due to the lack of time and computing resources, learning rate and batch size are not at the moment tuned to best values which can increase the performance of the model.
![loss_graph](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/loss_graph.png)

#### Result on some Test Images
This image below shows detection of a stop sign, As we can see the detection bounding box is not perfect but still our model has successfully classfied it with some error in the position in the image.
![stop_sign](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/stop_sign.png)

In the following, the model has successfully defined the class i.e person and IoU score is also shown which tells there is a mismatch in the ground truth and our model's position.

![girl](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/girl.png)

Below is an image of Zebra, the model detects the zebra and localizes it even though it fails to localize it accurately. 
![Figure_1](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/Figure_1.png)

Here, 3 persons are standing in the image, the model successully predicts that the objects in the image are persons but it fails to localize them with great accuracy. The model gives 4 bounding boxes for 3 persons. 
![Figure_2](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/Figure_2.png)

#### Class Prediction vs. Ground Truth Table

| **Image Numbrer** | **Predicted Class** | **Ground Truth Class** |
|-------------------|---------------------|------------------------|
| 1                 | person              | person                 |
| 1                 | tie                 | N/A                    |
| 2                 | stop sign           | stop sign              |
| 3                 | Person              | Person                 |
| 3                 | Person              | Person                 |
| 3                 | Person              | Person                 |
| 4                 | Zebra               | Zebra                  |

--

#### Custom Model Precision and Recall over multiple IoU Thresholds
The figures below shows Precision and Recall for different IoU thresholds. As the model didn't perform much better, so a very low IoU threshold has been used.

![Figure_3](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/Precision_custom_model.png)

![Figure_4](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/Recall_custom_model.png)

### FasterRCNN
We have also worked on FasterRCNN to detect the objects along with the class predicition. The below image shows the results of FasterRCNN used for object detection.
![fasterRCNN](https://github.com/EhtishamAshraf/Object_Detection_Models/blob/b89d80f34036d1c1e63338c74378359bc362d081/img/Faster_rcnn.png)


### How to run
The repository contains two notebooks:
1. Yolov5__VF.ipynb: 
This notebook contains the code for yolov5s, yolov5l, yolov5m, yolov5n.
The code can be run by uploading the notebook on "google colab" and also uploading the cocomini dataset.
The currently selected model is "yolov5s", but other models can be tested by changing the model's name in the notebook.

2. Object_Detection_Model.ipynb:
This notebook contains the code for our custom object detection model.
The code can be run by uploading the notebook on "kaggle" and importing coco2017 dataset.

3. iou.py:
This python file contains the code for calculating the IoU for the custom model.

4. faster_rcnn.py:
This python file contains the code for detecting the objects using FasterRCNN.

### References
1. https://docs.ultralytics.com/yolov5/
2. https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
