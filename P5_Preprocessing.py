import os
from P2_tick_scanner import processTicks
from P4_axis_tick_scanner import processBoundingBoxes
from P2_Test_OCR import get_model as get_tick_model
from P4_axis_tick_detector import get_model as get_axis_model
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compute_average_box_size(boxes):
    total_width = 0
    total_height = 0
    for box in boxes:
        width = box[0][2] - box[0][0]
        height = box[0][3] - box[0][1]
        total_width += width
        total_height += height
    avg_width = total_width / len(boxes) if boxes else 0
    avg_height = total_height / len(boxes) if boxes else 0
    return avg_width, avg_height

def compute_overlap(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    overlap = max(0, end - start)
    percentage1 = overlap / (interval2[1] - interval2[0])
    percentage2 = overlap / (interval1[1] - interval1[0])
    return max(percentage1, percentage2)

def is_match(tick_box, axis_box, axis_type, threshold=0.4):
    if axis_type == 'x':
        overlap = compute_overlap((tick_box[0][0], tick_box[0][2]), (axis_box[0][0], axis_box[0][2]))
    elif axis_type == 'y':
        overlap = compute_overlap((tick_box[0][1], tick_box[0][3]), (axis_box[0][1], axis_box[0][3]))
    else:
        return False, 0
    return overlap >= threshold, overlap

def compute_center_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def plot_boxes_and_points(image_path, matched_boxes):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(np.array(image))

    # Plot boxes and points
    for tick_box, ocr, axis_box, axis_type in matched_boxes:
        # Plot tick box
        rect_tick = patches.Rectangle((tick_box[0], tick_box[1]), tick_box[2] - tick_box[0], tick_box[3] - tick_box[1],
                                      linewidth=2, edgecolor='g' if axis_type == 'y' else 'b', facecolor='none')
        ax.add_patch(rect_tick)

        # Convert axis box to axis point and plot
        axis_center = ((axis_box[0] + axis_box[2]) / 2, (axis_box[1] + axis_box[3]) / 2)
        ax.plot(axis_center[0], axis_center[1], 'go' if axis_type == 'y' else 'bo')

        # Plot OCR text
        ax.text(tick_box[0], tick_box[1], ocr, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

def is_numerical(ocr_list):
    try:
        for ocr in ocr_list:
            float(ocr)
        return True
    except ValueError:
        return False

def calculate_value_per_pixel(pairs):
    if len(pairs) < 2:
        return None
    (coord1, ocr1), (coord2, ocr2) = pairs[0], pairs[1]
    try:
        ocr1, ocr2 = float(ocr1), float(ocr2)
        return abs(ocr2 - ocr1) / abs(coord2 - coord1)
    except ValueError:
        return None

def get_sample_point(pairs):
    if len(pairs) == 0:
        return None
    return pairs[0]

def process_image(image_path):
    print(f"Processing {image_path}")

    # Process tick labels
    tick_boxes = processTicks(image_path, visualize=True)

    # Process axis labels
    axis_model = get_axis_model(num_classes=2)  # 2 classes: background and bar
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    axis_model.load_state_dict(torch.load('tick_detector_500.pth', map_location=device))
    axis_model.to(device)
    axis_boxes = processBoundingBoxes(image_path, axis_model, device, visualize=True)

    # Separate x and y labels
    y_tick_boxes = [box for box in tick_boxes if box[2] == 'y']
    x_tick_boxes = [box for box in tick_boxes if box[2] == 'x']
    y_axis_boxes = [box for box in axis_boxes if box[1] == 'y']
    x_axis_boxes = [box for box in axis_boxes if box[1] == 'x']

    # Compute average box size
    avg_x_tick_size = compute_average_box_size(x_tick_boxes)
    avg_y_tick_size = compute_average_box_size(y_tick_boxes)
    avg_x_axis_size = compute_average_box_size(x_axis_boxes)
    avg_y_axis_size = compute_average_box_size(y_axis_boxes)

    # Find and print matched pairs for x and y
    def find_best_matches(tick_boxes, axis_boxes, axis_type):
        matches = []
        remaining_tick_boxes = tick_boxes.copy()
        remaining_axis_boxes = axis_boxes.copy()

        for tick_box in tick_boxes:
            best_match = None
            highest_overlap = 0
            for axis_box in axis_boxes:
                match, overlap = is_match(tick_box, axis_box, axis_type, threshold=0.4)
                if match and overlap > highest_overlap:
                    highest_overlap = overlap
                    best_match = axis_box
            if best_match:
                matches.append((tick_box, best_match, highest_overlap))
                remaining_tick_boxes = [box for box in remaining_tick_boxes if not np.array_equal(box[0], tick_box[0])]
                remaining_axis_boxes = [box for box in remaining_axis_boxes if not np.array_equal(box[0], best_match[0])]

        return matches, remaining_tick_boxes, remaining_axis_boxes

    x_matches, remaining_x_tick_boxes, remaining_x_axis_boxes = find_best_matches(x_tick_boxes, x_axis_boxes, 'x')
    y_matches, remaining_y_tick_boxes, remaining_y_axis_boxes = find_best_matches(y_tick_boxes, y_axis_boxes, 'y')

    # Combine the matched pairs into 4-tuples and compute distances
    complete_matches = [(tick[0], tick[1], axis[0], 'x') for tick, axis, _ in x_matches] + \
                       [(tick[0], tick[1], axis[0], 'y') for tick, axis, _ in y_matches]

    # Calculate the distances between the centers of matched tick and axis boxes
    distances = []
    x_distances = []
    y_distances = []

    for match in complete_matches:
        tick_box, ocr, axis_box, axis_type = match
        distance = compute_center_distance(tick_box, axis_box)
        distances.append(distance)
        if axis_type == 'x':
            x_distances.append(distance)
        elif axis_type == 'y':
            y_distances.append(distance)

    # Compute average distances
    avg_x_distance = np.mean(x_distances) if x_distances else 0
    avg_y_distance = np.mean(y_distances) if y_distances else 0

    # Create incomplete 4-tuples for remaining tick and axis boxes
    incomplete_matches = [(box[0], box[1], None, 'x') for box in remaining_x_tick_boxes] + \
                         [(box[0], box[1], None, 'y') for box in remaining_y_tick_boxes] + \
                         [(None, None, box[0], 'x') for box in remaining_x_axis_boxes] + \
                         [(None, None, box[0], 'y') for box in remaining_y_axis_boxes]

    # Fill in the incomplete 4-tuples
    filled_in_matches = []
    for match in incomplete_matches:
        tickbox, ocr, axisbox, axis_type = match
        if tickbox is not None and axisbox is None:
            # Form (tickbox, ocr, None, type)
            tick_center = ((tickbox[0] + tickbox[2]) / 2, (tickbox[1] + tickbox[3]) / 2)
            if axis_type == 'x':
                new_center = (tick_center[0], tick_center[1] - avg_x_distance)
                predicted_axisbox = [new_center[0] - avg_x_axis_size[0] / 2, new_center[1] - avg_x_axis_size[1] / 2,
                                     new_center[0] + avg_x_axis_size[0] / 2, new_center[1] + avg_x_axis_size[1] / 2]
            elif axis_type == 'y':
                new_center = (tick_center[0] + avg_y_distance, tick_center[1])
                predicted_axisbox = [new_center[0] - avg_y_axis_size[0] / 2, new_center[1] - avg_y_axis_size[1] / 2,
                                     new_center[0] + avg_y_axis_size[0] / 2, new_center[1] + avg_y_axis_size[1] / 2]
            filled_in_matches.append((tickbox, ocr, predicted_axisbox, axis_type))
        elif axisbox is not None and tickbox is None:
            # Form (None, None, axisbox, type)
            axis_center = ((axisbox[0] + axisbox[2]) / 2, (axisbox[1] + axisbox[3]) / 2)
            if axis_type == 'x':
                new_center = (axis_center[0], axis_center[1] + avg_x_distance)
                predicted_tickbox = [new_center[0] - avg_x_tick_size[0] / 2, new_center[1] - avg_x_tick_size[1] / 2,
                                     new_center[0] + avg_x_tick_size[0] / 2, new_center[1] + avg_x_tick_size[1] / 2]
            elif axis_type == 'y':
                new_center = (axis_center[0] - avg_y_distance, axis_center[1])
                predicted_tickbox = [new_center[0] - avg_y_tick_size[0] / 2, new_center[1] - avg_y_tick_size[1] / 2,
                                     new_center[0] + avg_y_tick_size[0] / 2, new_center[1] + avg_y_tick_size[1] / 2]
            filled_in_matches.append((predicted_tickbox, "AUTOFILLED", axisbox, axis_type))

    # Add the filled tuples to the list of complete tuples
    complete_matches.extend(filled_in_matches)

    # Determine axis categories
    x_ocr_list = [match[1] for match in complete_matches if match[3] == 'x']
    y_ocr_list = [match[1] for match in complete_matches if match[3] == 'y']

    x_is_numerical = is_numerical(x_ocr_list)
    y_is_numerical = is_numerical(y_ocr_list)

    x_axis_category = 'Numerical' if x_is_numerical else 'Categorical'
    y_axis_category = 'Numerical' if y_is_numerical else 'Categorical'

    # Calculate value per pixel and sample points
    x_pairs = [(int((match[2][0] + match[2][2]) / 2), float(match[1])) for match in complete_matches if match[3] == 'x' and x_is_numerical]
    y_pairs = [(int((match[2][1] + match[2][3]) / 2), float(match[1])) for match in complete_matches if match[3] == 'y' and y_is_numerical]

    x_value_per_pixel = calculate_value_per_pixel(x_pairs) if x_is_numerical else None
    y_value_per_pixel = calculate_value_per_pixel(y_pairs) if y_is_numerical else None

    x_sample_point = get_sample_point(x_pairs) if x_is_numerical else None
    y_sample_point = get_sample_point(y_pairs) if y_is_numerical else None

    # Plot the boxes and points
    plot_boxes_and_points(image_path, complete_matches)

    # Return the results
    return (os.path.splitext(os.path.basename(image_path))[0], x_value_per_pixel, x_sample_point, y_value_per_pixel, y_sample_point)

# Directory containing images
image_dir = 'test/images'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

# List to store results
results = []

# Process each image
#for image_path in image_files:
#    result = process_image(image_path)
#    results.append(result)

# Print all results
#print("\nAll results:")
#for result in results:
#    print(result)
