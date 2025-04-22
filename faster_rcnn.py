# Importing all the libraries
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CocoDetection
from PIL import Image, ImageDraw
import numpy as np
import os
import random

# Making sure that the path to the folder exists:
os.makedirs("output", exist_ok=True)

# Using the same COCO class names as in the training code
COCO_INSTANCE_CATEGORY_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Defining the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading the validation dataset
print("Loading validation dataset...")
val_dataset_full = CocoDetection(
    root='./tiny_coco/val2017',
    annFile='./tiny_coco/annotations/instances_val2017.json',
    transform=transform
)

# We just used a random subset of the validation dataset
subset_percentage = 1  # 1% for testing
random.seed(42)
val_indices = random.sample(range(len(val_dataset_full)), int(len(val_dataset_full) * subset_percentage))
val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

# Data loader
def collate_fn(batch):
    images, targets = zip(*batch)
    valid_batch = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt) > 0]
    if len(valid_batch) == 0:
        return [], []
    images, targets = zip(*valid_batch)
    return list(images), list(targets)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

# Here we are loading the saved model
print("Loading the saved model...")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 81  # Number of classes during training (80 + 1 background)

# Defining the model:
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("model_epoch_9.pth", map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully.")

# This Function filters predictions with confidence > threshold
def filter_predictions(outputs, threshold=0.5):
    filtered_outputs = []
    for output in outputs:
        keep = output['scores'] > threshold
        filtered_outputs.append({
            'boxes': output['boxes'][keep],
            'labels': output['labels'][keep],
            'scores': output['scores'][keep],
        })
    return filtered_outputs

# This Function is used to draw predictions on the image
def draw_predictions(image, predictions):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        box = box.tolist()
        label_text = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        if label_text != "N/A":  # Skip N/A categories
            score_text = f"{score.item() * 100:.1f}%"
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label_text}: {score_text}", fill="yellow")
    return image

# This Function is used to denormalize the image
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

# Evaluvating and and saving the results
print("Starting evaluation...")
max_eval_batches = 5  # number of batches
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        if len(images) == 0:
            print(f"Skipping empty batch {batch_idx + 1}.")
            continue

        if batch_idx >= max_eval_batches:
            print("comes here")
            break

        for i, image_tensor in enumerate(images):
            normalized_image = image_tensor.cpu()
            denormalized_image = denormalize(normalized_image, mean, std).clamp(0, 1)
            original_image_pil = transforms.ToPILImage()(denormalized_image)

            image_tensor = image_tensor.to(device)
            outputs = model([image_tensor])
            filtered_output = filter_predictions(outputs, threshold=0.5)[0]

            processed_image_pil = original_image_pil.copy()  # Copy original image
            processed_image_pil = draw_predictions(processed_image_pil, filtered_output)

            # Saving the processed image
            processed_image_pil.save(f"output/image_{batch_idx * 4 + i + 1}.jpg")

            # Log detections
            print(f"Image {batch_idx * 4 + i + 1}:")
            for box, label, score in zip(filtered_output['boxes'], filtered_output['labels'], filtered_output['scores']):
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
                confidence = f"{score.item() * 100:.1f}%"
                print(f"  Detected: {label_name}, Confidence: {confidence}")
