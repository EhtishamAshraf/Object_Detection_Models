# Importing Libraries
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Object Detection Model
class CustomObjectDetectionModel(nn.Module):
    def __init__(self, num_classes, num_boxes_per_image):
        super(CustomObjectDetectionModel, self).__init__()
        self.num_boxes_per_image = num_boxes_per_image
        self.backbone = torchvision.models.resnet50(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove FC layers
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_boxes_per_image * num_classes)
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_boxes_per_image * 4)
        )
   # Forward Function
    def forward(self, x):
        features = self.backbone(x)
        pooled_features = nn.AdaptiveAvgPool2d((7, 7))(features)
        class_logits = self.classification_head(pooled_features)
        bbox_regressions = self.regression_head(pooled_features)
        batch_size = x.shape[0]
        class_logits = class_logits.view(batch_size, self.num_boxes_per_image, -1)
        bbox_regressions = bbox_regressions.view(batch_size, self.num_boxes_per_image, 4)
        return class_logits, bbox_regressions


# Function to calculate IoU for the given images
def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


# Loading custom pre-trained model
model_path = "model_epoch_6.pth"
num_classes = len(COCO_INSTANCE_CATEGORY_NAMES)
num_boxes_per_image = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomObjectDetectionModel(num_classes=num_classes, num_boxes_per_image=num_boxes_per_image).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # evaluating the model

# Resizing, transforming and normalizing the input image
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading the labeled data in COCO format
annotations_path = "/home/roboticslab/Downloads/AI_data/tiny_coco/annotations/instances_val2017.json"
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Iterating through input images
images_path = "tiny_coco/val2017"
image_files = os.listdir(images_path)[:10]  # Limit to 10 images for demonstration

def draw_predictions_and_ground_truth(image, predicted_boxes, ground_truth_boxes, labels, scores, iou_values, threshold=0.8):
    draw = ImageDraw.Draw(image)
    
    # Drawing the predicted Bboxes
    for box, label, score, iou in zip(predicted_boxes, labels, scores, iou_values):
        if score > threshold:  # Confidence threshold
            box = box.tolist()
            label_text = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
            score_text = f"{score.item() * 100:.1f}%, IoU: {iou:.2f}"
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label_text}: {score_text}", fill="yellow")

    # Drawing the ground-truth Bboxes
    for gt_box in ground_truth_boxes:
        gt_box = gt_box.tolist()
        draw.rectangle(gt_box, outline="green", width=2)  # Ground truth in green

    return image

# Iterating through the tiny coco-dataset 
for image_file in image_files:
    image_path = os.path.join(images_path, image_file)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, bbox_regressions = model(input_tensor)

    # Post-process outputs
    class_probs = torch.softmax(class_logits, dim=-1)
    top_scores, top_labels = class_probs[0].max(dim=-1)
    predicted_boxes = bbox_regressions[0].cpu().numpy()

    # Convert predictions to [x_min, y_min, x_max, y_max] format
    predicted_boxes[:, 2] += predicted_boxes[:, 0]
    predicted_boxes[:, 3] += predicted_boxes[:, 1]

    # Get ground-truth annotations for the current image
    image_id = int(image_file.split('.')[0])  # Assuming image IDs are part of filenames
    gt_annotations = [
        ann for ann in annotations['annotations'] if ann['image_id'] == image_id
    ]

    if len(gt_annotations) == 0:
        print(f"No ground-truth annotations for image {image_file}.")
        continue

    # Extract and reshape ground-truth bounding boxes
    gt_boxes = np.array([ann['bbox'] for ann in gt_annotations], dtype=np.float32)
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes.reshape(-1, 4)

    # Converting the width and height to x_max and y_max
    gt_boxes[:, 2] += gt_boxes[:, 0]  
    gt_boxes[:, 3] += gt_boxes[:, 1]  

    # Calculating the IoU and writing on the IMage
    iou_values = []
    for pred_box in predicted_boxes:
        max_iou = 0
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)  
        iou_values.append(max_iou)

    # Drawing predictions, ground truth, and IoU on the image
    output_image = draw_predictions_and_ground_truth(image, predicted_boxes, gt_boxes, top_labels, top_scores, iou_values, threshold=0.8)

    # Displaying the image with predictions and ground truth
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()
