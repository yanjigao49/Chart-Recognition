from P2_Test_OCR import (
    get_model, ChartDataset, collate_fn, transform,
    apply_nms, group_boxes, expand_box, ocr_on_expanded_boxes,
    process_values, correct_values, is_number, compute_most_frequent_difference
)
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def process_values(values):
    # Replace non-numeric values with 0 and convert to float
    numeric_values = [float(v) if is_number(v) else 0 for v in values]
    if len(numeric_values) >= 2:
        corrected_values = correct_values(numeric_values)
        return corrected_values
    return numeric_values

def processTicks(image_path, visualize=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load('axis_bounding_box_detector_500.pth'))
    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
    output = {k: v.to(torch.device("cpu")) for k, v in outputs[0].items()}
    output = apply_nms(output)

    y_tick_groups, x_tick_groups = group_boxes(output)
    results = []

    for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
        box_np = box.detach().cpu().numpy()
        x1, y1, x2, y2 = box_np

        if any(np.array_equal(box_np, g) for group in y_tick_groups for g in group):
            color = 'g'
        elif any(np.array_equal(box_np, g) for group in x_tick_groups for g in group):
            color = 'b'
        else:
            continue

        final_ocr_result = None
        expand_pixels = 5
        expanded_box = expand_box(box_np, expand_pixels, image.size[::-1])
        ex1, ey1, ex2, ey2 = expanded_box

        box_img = np.array(image.crop(expanded_box))
        box_img = (box_img * 255).astype(np.uint8)
        ocr_results = ocr_on_expanded_boxes(box_img)

        if final_ocr_result is None or not is_number(final_ocr_result):
            final_ocr_result = ocr_results['final_ocr']

        result_type = 'y' if color == 'g' else 'x'
        results.append(((x1, y1, x2, y2), final_ocr_result, result_type))

    # Sort results based on the center of the bounding box
    results_y_sorted = sorted((r for r in results if r[2] == 'y'), key=lambda r: (r[0][1] + r[0][3]) / 2, reverse=True)
    results_x_sorted = sorted((r for r in results if r[2] == 'x'), key=lambda r: (r[0][0] + r[0][2]) / 2)

    y_values_sorted = [r[1] for r in results_y_sorted]
    x_values_sorted = [r[1] for r in results_x_sorted]

    print("Before correction:")
    print("X values:", x_values_sorted)
    print("Y values:", y_values_sorted)

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

    print("After correction:")
    print("Corrected X values:", corrected_x_values)
    print("Corrected Y values:", corrected_y_values)

    # Ensure the length of corrected values match the original sorted lists
    corrected_y_values = corrected_y_values[:len(results_y_sorted)]
    corrected_x_values = corrected_x_values[:len(results_x_sorted)]

    for i in range(len(results_y_sorted)):
        results_y_sorted[i] = (results_y_sorted[i][0], corrected_y_values[i], results_y_sorted[i][2])

    for i in range(len(results_x_sorted)):
        results_x_sorted[i] = (results_x_sorted[i][0], corrected_x_values[i], results_x_sorted[i][2])

    results = results_y_sorted + results_x_sorted

    if visualize:
        label_map = {0: 'background', 1: 'y_tick_label', 2: 'x_tick_label'}
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image_np, cmap='gray')

        for box, ocr_text, result_type in results:
            x1, y1, x2, y2 = box
            color = 'g' if result_type == 'y' else 'b'
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y2 + 10, ocr_text, color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.show()

    return results

#image_path = 'test/images/01b45b831589.jpg'
#boxes = processTicks(image_path, visualize=True)
#for box in boxes:
#    print(box)
