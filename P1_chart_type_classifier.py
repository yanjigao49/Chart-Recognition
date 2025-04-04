import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from timm import create_model
from PIL import Image
from sklearn.model_selection import train_test_split

def filter_samples(imgs, annots, num_samples, chart_types, root_dir):
    chart_dict = {chart_type: [] for chart_type in chart_types}
    for img, annot in zip(imgs, annots):
        annot_path = os.path.join(root_dir, "annotations", annot)
        with open(annot_path) as f:
            data = json.load(f)
            chart_type = data['chart-type']
            if chart_type in chart_dict:
                img_path = os.path.join(root_dir, "images", img)
                chart_dict[chart_type].append((img_path, annot_path))

    samples_per_type = num_samples // len(chart_types)
    extra_samples = num_samples % len(chart_types)

    final_imgs = []
    final_annots = []
    type_counts = {chart_type: 0 for chart_type in chart_types}
    for chart_type in chart_types:
        samples = chart_dict[chart_type][:samples_per_type + (1 if extra_samples > 0 else 0)]
        extra_samples = max(0, extra_samples - 1)
        final_imgs.extend([img for img, _ in samples])
        final_annots.extend([annot for _, annot in samples])
        type_counts[chart_type] += len(samples)

    for chart_type, count in type_counts.items():
        print(f"{chart_type}: {count}")

    return final_imgs, final_annots, type_counts

class ChartDataset(Dataset):
    def __init__(self, images, annotations=None, chart_types=None, transform=None):
        self.images = images
        self.labels = []
        self.chart_types = chart_types
        self.transform = transform

        if annotations:
            for annotation_path in annotations:
                annotation = self.load_annotations(annotation_path)
                self.labels.append(annotation)

    def load_image(self, image_path):
        image = Image.open(image_path)  # Load image using PIL
        if self.transform:
            image = self.transform(image)
        return image

    def load_annotations(self, annotations_path):
        with open(annotations_path, 'r') as file:
            annotation = json.load(file)
        label = self.chart_types.index(annotation['chart-type'])
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = self.load_image(image_path)
        if self.labels:
            label = self.labels[idx]
            return image, label
        else:
            return image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

chart_types = ['dot', 'line', 'scatter', 'vertical_bar', 'horizontal_bar']
num_classes = len(chart_types)
num_samples = 365

# Define the root directory for images and annotations
root_dir = 'train'

# Get the list of images and annotations
images_dir = os.path.join(root_dir, 'images')
annotations_dir = os.path.join(root_dir, 'annotations')

image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
annotation_files = [f.replace('.jpg', '.json') for f in image_files]

# Filter the samples
filtered_imgs, filtered_annots, type_counts = filter_samples(image_files, annotation_files, num_samples, chart_types, root_dir)

# Create the dataset
dataset = ChartDataset(filtered_imgs, filtered_annots, chart_types, transform=transform)

# Extract labels for stratified splitting
labels = [dataset.load_annotations(ann) for ann in filtered_annots]

# Stratified split into training and remaining data
train_imgs, remaining_imgs, train_labels, remaining_labels = train_test_split(
    filtered_imgs, labels, stratify=labels, test_size=0.2, random_state=42
)

# Further stratified split remaining data into validation and test sets
val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    remaining_imgs, remaining_labels, stratify=remaining_labels, test_size=0.5, random_state=42
)

# Create datasets
train_dataset = ChartDataset(train_imgs, [filtered_annots[filtered_imgs.index(img)] for img in train_imgs], chart_types, transform=transform)
val_dataset = ChartDataset(val_imgs, [filtered_annots[filtered_imgs.index(img)] for img in val_imgs], chart_types, transform=transform)
test_dataset = ChartDataset(test_imgs, [filtered_annots[filtered_imgs.index(img)] for img in test_imgs], chart_types, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

# Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final layer
for param in model.head.parameters():
    param.requires_grad = True

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=0.001)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'chart_type_classifier_365.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = torch.tensor(labels).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

train_model(model, train_loader, val_loader, criterion, optimizer)

model.load_state_dict(torch.load('chart_type_classifier.pth'))

evaluate_model(model, test_loader, criterion)

def predict_chart_types(model, test_loader, chart_types):
    model.eval()
    predicted_chart_types = []

    with torch.no_grad():
        for images in test_loader:
            images = [image.to(device) for image in images]
            images = torch.stack(images)  # Stack images to form a batch
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_chart_types.extend([chart_types[label] for label in predicted.cpu().numpy()])

    for image_path, chart_type in zip(test_dataset.dataset.images, predicted_chart_types):
        print(f"Image: {image_path}, Predicted Chart Type: {chart_type}")

predict_chart_types(model, test_loader, chart_types)
