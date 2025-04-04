import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pytesseract
import cv2
import numpy as np
from collections import Counter

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to enhance text clarity using unsharp masking
def enhance_text_clarity_unsharp_masking(image):
    gray = np.array(image)
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (9, 9), 2.0)
    sharpened_image = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    return Image.fromarray(sharpened_image)


class ChartDataset(torch.utils.data.Dataset):
    def __init__(self, root, num_samples=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        if num_samples is not None:
            self.imgs = self.imgs[:num_samples]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("L")
        target = {}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def transform(img):
    img = F.to_tensor(img)
    return img


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def apply_nms(output, iou_threshold=0.35, score_threshold=0.70):
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    indices = nms(boxes, scores, iou_threshold)
    nms_boxes = boxes[indices]
    nms_scores = scores[indices]
    nms_labels = labels[indices]

    keep = nms_scores >= score_threshold
    nms_boxes = nms_boxes[keep]
    nms_scores = nms_scores[keep]
    nms_labels = nms_labels[keep]

    return {'boxes': nms_boxes, 'scores': nms_scores, 'labels': nms_labels}


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def group_boxes(output, threshold=5):
    y_tick_groups = []
    x_tick_groups = []

    boxes = output['boxes'].detach().cpu().numpy()
    matched_boxes = np.zeros(len(boxes), dtype=bool)

    print(f"Initial boxes: {len(boxes)}")

    for i, box in enumerate(boxes):
        if matched_boxes[i]:
            continue
        x1, y1, x2, y2 = box
        print(f"Checking box {i}: {box}")

        current_y_group = [box]
        current_x_group = [box]

        for j, other_box in enumerate(boxes):
            if i == j or matched_boxes[j]:
                continue
            ox1, oy1, ox2, oy2 = other_box

            if abs(x2 - ox2) <= threshold:
                print(f"Matching vertical box {i} with box {j}: {other_box}")
                current_y_group.append(other_box)
                matched_boxes[j] = True

            if abs(y1 - oy1) <= threshold:
                print(f"Matching horizontal box {i} with box {j}: {other_box}")
                current_x_group.append(other_box)
                matched_boxes[j] = True

        if len(current_y_group) > 1:
            y_tick_groups.append(current_y_group)
        if len(current_x_group) > 1:
            x_tick_groups.append(current_x_group)

        matched_boxes[i] = True

    print(f"Y-tick groups: {len(y_tick_groups)}")
    print(f"X-tick groups: {len(x_tick_groups)}")

    return y_tick_groups, x_tick_groups


def expand_box(box, expand_pixels, image_shape):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - expand_pixels)
    y1 = max(0, y1 - expand_pixels)
    x2 = min(image_shape[1], x2 + expand_pixels)
    y2 = min(image_shape[0], y2 + expand_pixels)
    return [x1, y1, x2, y2]


def ocr_on_expanded_boxes(box_img):
    original_ocr = pytesseract.image_to_string(cv2.cvtColor(box_img, cv2.COLOR_GRAY2BGR), config='--psm 6').strip()
    unsharp_img = enhance_text_clarity_unsharp_masking(Image.fromarray(box_img.squeeze()))
    unsharp_ocr = pytesseract.image_to_string(cv2.cvtColor(np.array(unsharp_img), cv2.COLOR_GRAY2BGR),
                                              config='--psm 6').strip()

    final_ocr = unsharp_ocr if is_number(unsharp_ocr) else (original_ocr if is_number(original_ocr) else unsharp_ocr)

    return {
        'original_ocr': original_ocr,
        'unsharp_ocr': unsharp_ocr,
        'final_ocr': final_ocr
    }


def compute_most_frequent_difference(values):
    differences = [round(values[i + 1] - values[i], 10) for i in range(len(values) - 1)]
    if not differences:
        return 0.0
    most_common = Counter(differences).most_common()
    if most_common[0][0] == 0 and len(most_common) > 1:
        second_common_diff = most_common[1][0]
        return (most_common[0][0] + second_common_diff) / 2
    return most_common[0][0]


def correct_values(values):
    most_frequent_diff = compute_most_frequent_difference(values)
    corrected_values = [values[0]]
    for i in range(1, len(values)):
        expected_value = corrected_values[-1] + most_frequent_diff
        if abs(expected_value - values[i]) > abs(most_frequent_diff) * 0.1:
            corrected_values.append(round(expected_value, 10))
        else:
            corrected_values.append(values[i])
    return corrected_values


def process_values(values):
    numeric_values = [float(v) for v in values if is_number(v)]
    if len(numeric_values) >= 2:
        corrected_values = correct_values(numeric_values)
        return corrected_values
    return values


def visualize_predictions(image, output, expand_pixels_list=[5, 6]):
    label_map = {0: 'background', 1: 'y_tick_label', 2: 'x_tick_label'}
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image, cmap='gray')

    y_tick_groups, x_tick_groups = group_boxes(output)

    x_values = []
    y_values = []
    x_boxes = []
    y_boxes = []

    for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
        box = box.detach().cpu().numpy()
        x1, y1, x2, y2 = box

        # Color boxes based on groupings
        if any(np.array_equal(box, g) for group in y_tick_groups for g in group):
            color = 'g'
            y_boxes.append(box)
        elif any(np.array_equal(box, g) for group in x_tick_groups for g in group):
            color = 'b'
            x_boxes.append(box)
        else:
            continue

        final_ocr_result = None

        for expand_pixels in expand_pixels_list:
            expanded_box = expand_box(box, expand_pixels, image.shape)
            ex1, ey1, ex2, ey2 = expanded_box

            rect = patches.Rectangle((ex1, ey1), ex2 - ex1, ey2 - ey1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(ex1, ey1 - 10, f'{score:.2f}', color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            label_name = label_map[label.item()]
            print(f"Predicted box: {[x1, y1, x2, y2]}, Confidence: {score:.2f}, Type: {label_name}")

            box_img = image[int(ey1):int(ey2), int(ex1):int(ex2)]
            box_img = (box_img * 255).astype(np.uint8)

            if len(box_img.shape) == 2:
                box_img = np.expand_dims(box_img, axis=-1)

            ocr_results = ocr_on_expanded_boxes(box_img)

            print(f'OCR Result (Expanded {expand_pixels} pixels) - Original: {ocr_results["original_ocr"]}')
            print(f'OCR Result (Expanded {expand_pixels} pixels) - Unsharp: {ocr_results["unsharp_ocr"]}')
            print(f'Final OCR Result (Expanded {expand_pixels} pixels): {ocr_results["final_ocr"]}')

            if final_ocr_result is None or not is_number(final_ocr_result):
                final_ocr_result = ocr_results['final_ocr']

        print(f'Final OCR Result for box: {final_ocr_result}')

        if any(np.array_equal(box, g) for group in y_tick_groups for g in group):
            y_values.append(final_ocr_result)
        elif any(np.array_equal(box, g) for group in x_tick_groups for g in group):
            x_values.append(final_ocr_result)

    x_values_sorted = [val for _, val in sorted(zip(x_boxes, x_values), key=lambda pair: pair[0][0])]
    y_values_sorted = [val for _, val in sorted(zip(y_boxes, y_values), key=lambda pair: pair[0][1], reverse=True)]

    print(f'OCR X Values: {x_values_sorted}')
    print(f'OCR Y Values: {y_values_sorted}')

    # Check if 75% or more are strings, if so, skip correction
    def should_skip(values):
        return (len([v for v in values if not is_number(v)]) / len(values)) >= 0.75

    if not should_skip(x_values_sorted):
        corrected_x_values = process_values(x_values_sorted)
    else:
        corrected_x_values = x_values_sorted

    if not should_skip(y_values_sorted):
        corrected_y_values = process_values(y_values_sorted)
    else:
        corrected_y_values = y_values_sorted

    print(f'Corrected OCR X Values: {corrected_x_values}')
    print(f'Corrected OCR Y Values: {corrected_y_values}')

    plt.show()


if __name__ == '__main__':
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load('axis_bounding_box_detector_500.pth'))
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    dataset = ChartDataset(root='test', num_samples=7, transforms=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for images, _ in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]

        for image, output in zip(images, outputs):
            output = apply_nms(output)
            visualize_predictions(image, output)

