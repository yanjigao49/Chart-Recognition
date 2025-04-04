from P4_axis_tick_test import (
    get_model, transform, apply_nms, visualize_predictions
)
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def processBoundingBoxes(image_path, model, device, visualize=False):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)[0]
        output = apply_nms(output)

    # Visualize predictions if requested
    if visualize:
        visualize_colored_predictions(img_tensor[0], output)

    # Group boxes
    y_tick_groups, x_tick_groups = group_boxes(output)

    # Prepare results
    boxes = []
    for box in output['boxes']:
        box_np = box.detach().cpu().numpy()
        if any(np.array_equal(box_np, g) for group in y_tick_groups for g in group):
            box_type = 'y'
        elif any(np.array_equal(box_np, g) for group in x_tick_groups for g in group):
            box_type = 'x'
        else:
            continue
        boxes.append((box_np, box_type))

    return boxes


def group_boxes(output, threshold=5):
    y_tick_groups = []
    x_tick_groups = []

    boxes = output['boxes'].detach().cpu().numpy()
    matched_boxes = np.zeros(len(boxes), dtype=bool)

    for i, box in enumerate(boxes):
        if matched_boxes[i]:
            continue
        x1, y1, x2, y2 = box

        current_y_group = [box]
        current_x_group = [box]

        for j, other_box in enumerate(boxes):
            if i == j or matched_boxes[j]:
                continue
            ox1, oy1, ox2, oy2 = other_box

            if abs(x2 - ox2) <= threshold:
                current_y_group.append(other_box)
                matched_boxes[j] = True

            if abs(y1 - oy1) <= threshold:
                current_x_group.append(other_box)
                matched_boxes[j] = True

        if len(current_y_group) > 1:
            y_tick_groups.append(current_y_group)
        if len(current_x_group) > 1:
            x_tick_groups.append(current_x_group)

        matched_boxes[i] = True

    return y_tick_groups, x_tick_groups


def visualize_colored_predictions(image, output):
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Group boxes
    y_tick_groups, x_tick_groups = group_boxes(output)

    # Plot predicted boxes with different colors for x and y
    for box in output['boxes']:
        box = box.detach().cpu().numpy()
        color = 'g' if any(np.array_equal(box, g) for group in y_tick_groups for g in group) else 'b'
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()


# Example usage
if __name__ == '__main__':
    model = get_model(num_classes=2)  # 2 classes: background and bar
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load('tick_detector_500.pth', map_location=device))
    model.to(device)

    #image_path = 'test/images/01b45b831589.jpg'
    #boxes = processBoundingBoxes(image_path, model, device, visualize=True)
    #for box in boxes:
    #    print(box)