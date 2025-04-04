import os
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def apply_nms(output, iou_threshold=0.2, score_threshold=0.96):
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


def visualize_predictions(image, output, save_path=None):
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Plot predicted boxes in red
    for box, score in zip(output['boxes'], output['scores']):
        box = box.detach().cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 10, f'{score:.2f}', color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    if save_path:
        plt.savefig(save_path)
    plt.show()


def load_and_predict(model, device, image_dir, save_visualizations=False):
    model.eval()
    images = list(sorted(os.listdir(image_dir)))
    results = []

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)[0]
            output = apply_nms(output)

        results.append((img_name, output))

        if save_visualizations:
            visualize_predictions(img_tensor[0], output)

        print(f"Image: {img_name}")
        for box in output['boxes']:
            print(f"Box: {box.detach().cpu().numpy()}")

    return results


if __name__ == '__main__':
    # Load the model and weights
    model = get_model(num_classes=2)  # 2 classes: background and bar
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load('tick_detector_500.pth', map_location=device))
    model.to(device)

    # Run prediction on test images and visualize results
    test_image_dir = 'test/images'
    load_and_predict(model, device, test_image_dir, save_visualizations=True)
