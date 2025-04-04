import os
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms


def transform(img):
    img = F.to_tensor(img)
    return img


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def apply_nms(output, iou_threshold=0.5, score_threshold=0.5):
    boxes = output['boxes']
    scores = output['scores']
    indices = nms(boxes, scores, iou_threshold)
    nms_boxes = boxes[indices]
    nms_scores = scores[indices]
    nms_labels = output['labels'][indices]

    # Filter out boxes with scores below threshold
    keep = nms_scores >= score_threshold
    nms_boxes = nms_boxes[keep]
    nms_scores = nms_scores[keep]
    nms_labels = nms_labels[keep]

    return {'boxes': nms_boxes, 'scores': nms_scores, 'labels': nms_labels}


def visualize_predictions(image, center_points, save_path=None):
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Plot predicted center points in red
    for center in center_points:
        ax.plot(center[0], center[1], 'ro')
        ax.text(center[0], center[1] - 10, f'{center[0]:.2f}, {center[1]:.2f}', color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    if save_path:
        plt.savefig(save_path)
    plt.show()


def predict_dot(image_path, model_weight_path='dot_detector_20.pth', iou_threshold=0.5, score_threshold=0.5,
                num_classes=2, save_visualization=False, save_path=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and transform the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        output = model(img_tensor)[0]
        output = apply_nms(output, iou_threshold, score_threshold)

    # Convert boxes to center points
    center_points = []
    for box in output['boxes']:
        box = box.detach().cpu().numpy()
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        center_points.append((center_x, center_y))

    # Visualize predictions if required
    if save_visualization:
        visualize_predictions(img_tensor[0], center_points, save_path)

    # Print results
    print(f"Image: {image_path}")
    for center in center_points:
        print(f"Center: {center}")

    return center_points


# Example usage
#image_path = 'test/images/0a2a7c40e9a6.jpg'
#output = predict_dot(image_path, save_visualization=True)
#print(output)
