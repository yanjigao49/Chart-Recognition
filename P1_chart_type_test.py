import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from timm import create_model
from PIL import Image


# Function to predict chart type for a single image
def predict_chart_type(image_path, transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]), model_weight_path = "chart_type_classifier_365.pth", chart_types = ['dot', 'line', 'scatter', 'vertical_bar', 'horizontal_bar']):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    # Load the model
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=len(chart_types))
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    model.to(device)

    # Make prediction
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_chart_type = chart_types[predicted.item()]

    return predicted_chart_type


# Example usage
#image_path = "test/images/00f5404753cf.jpg"

#predicted_chart_type = predict_chart_type(image_path)
#print(f"The predicted chart type is: {predicted_chart_type}")
