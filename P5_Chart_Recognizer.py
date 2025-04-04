import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from P5_Preprocessing import process_image
from P1_chart_type_test import predict_chart_type
from P3_bar_test import predict_bar
from P3_line_test import predict_line
from P3_dot_test import predict_dot
from P3_scatterplot_test import predict_scatter

def convert_to_value(coord, axis_value_per_pixel, axis_sample_point, is_y_axis=False):
    if axis_value_per_pixel is None or axis_sample_point is None:
        return None
    point_location, point_value = axis_sample_point
    if is_y_axis:
        pixel_distance = point_location - coord  # Flip the distance calculation for y-axis
    else:
        pixel_distance = coord - point_location
    value_distance = pixel_distance * axis_value_per_pixel
    return point_value + value_distance

def convert_dot_locations(dot_locations, x_info, y_info):
    x_axis_value_per_pixel, x_sample_point = x_info
    y_axis_value_per_pixel, y_sample_point = y_info
    value_pairs = []
    for x_pixel, y_pixel in dot_locations:
        x_value = convert_to_value(x_pixel, x_axis_value_per_pixel, x_sample_point)
        y_value = convert_to_value(y_pixel, y_axis_value_per_pixel, y_sample_point, is_y_axis=True)
        value_pairs.append((x_pixel, y_pixel, x_value, y_value))
    return value_pairs

def convert_bar_locations(bar_locations, x_info, y_info, is_vertical=True):
    x_axis_value_per_pixel, x_sample_point = x_info
    y_axis_value_per_pixel, y_sample_point = y_info
    value_pairs = []
    for bar in bar_locations['boxes']:
        x1, y1, x2, y2 = bar.cpu().numpy()
        if is_vertical:
            mid_x = (x1 + x2) / 2
            y_value = convert_to_value(y1, y_axis_value_per_pixel, y_sample_point, is_y_axis=True)
            value_pairs.append((mid_x, y1, y_value))
        else:
            mid_y = (y1 + y2) / 2
            x_value = convert_to_value(x2, x_axis_value_per_pixel, x_sample_point)
            value_pairs.append((x2, mid_y, x_value))
    return value_pairs

def group_dot_locations(dot_locations, threshold=10):
    groups = []
    while dot_locations:
        point = dot_locations.pop(0)
        group = [point]
        remaining_points = []
        for other_point in dot_locations:
            if abs(other_point[0] - point[0]) <= threshold:
                group.append(other_point)
            else:
                remaining_points.append(other_point)
        groups.append(group)
        dot_locations = remaining_points
    return groups

def analyze_dot_groups(dot_groups):
    analyzed_groups = []
    for group in dot_groups:
        highest_point = min(group, key=lambda x: x[1])  # Assuming lower y-values are higher on the chart
        count = len(group)
        analyzed_groups.append((highest_point, count))
    return analyzed_groups

def plot_dot_groups(image_path, analyzed_groups):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(np.array(image))

    for (x_pixel, y_pixel), count in analyzed_groups:
        ax.text(x_pixel, y_pixel, f'{count}', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        ax.plot(x_pixel, y_pixel, 'bo')

    plt.show()

def plot_value_pairs(image_path, value_pairs, is_vertical=True, is_dot_or_line=False):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(np.array(image))

    # Plot value pairs
    for pair in value_pairs:
        if is_dot_or_line:
            x_pixel, y_pixel, x_value, y_value = pair
            ax.text(x_pixel, y_pixel, f'({x_value:.2f}, {y_value:.2f})', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        else:
            coord1, coord2, value = pair
            ax.text(coord1, coord2, f'{value:.2f}', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

def chart_prediction(image_path):
    # 1. Predict the chart type
    chart_type = predict_chart_type(image_path)
    print(f"Chart Type: {chart_type}")

    # 2. Process the image to get the 5-tuple
    image_info = process_image(image_path)
    print(f"Image Info: {image_info}")
    file_name, x_axis_value_per_pixel, x_sample_point, y_axis_value_per_pixel, y_sample_point = image_info

    # Check if any element in the 5-tuple is None for line and scatter plots
    if chart_type in ["line", "scatter"] and any(info is None for info in image_info):
        print(f"Skipping '{file_name}' due to not enough info")
        return

    # Check for bar charts
    if chart_type == "vertical_bar" and y_axis_value_per_pixel is None:
        print(f"Skipping '{file_name}' due to not enough y-axis info")
        return

    if chart_type == "horizontal_bar" and x_axis_value_per_pixel is None:
        print(f"Skipping '{file_name}' due to not enough x-axis info")
        return

    # 3. Data recognition based on the chart type
    if chart_type == "line":
        dot_locations = predict_line(image_path)
        print(f"Dot Locations: {dot_locations}")
        value_pairs = convert_dot_locations(dot_locations, (x_axis_value_per_pixel, x_sample_point), (y_axis_value_per_pixel, y_sample_point))
        print(f"Value Pairs: {value_pairs}")
        plot_value_pairs(image_path, value_pairs, is_dot_or_line=True)

    elif chart_type == "scatter":
        dot_locations = predict_scatter(image_path)
        print(f"Dot Locations: {dot_locations}")
        value_pairs = convert_dot_locations(dot_locations, (x_axis_value_per_pixel, x_sample_point), (y_axis_value_per_pixel, y_sample_point))
        print(f"Value Pairs: {value_pairs}")
        plot_value_pairs(image_path, value_pairs, is_dot_or_line=True)

    elif chart_type == "vertical_bar":
        bar_locations = predict_bar(image_path)
        print(f"Bounding Boxes: {bar_locations}")
        value_pairs = convert_bar_locations(bar_locations, (x_axis_value_per_pixel, x_sample_point), (y_axis_value_per_pixel, y_sample_point), is_vertical=True)
        print(f"Value Pairs: {value_pairs}")
        plot_value_pairs(image_path, value_pairs, is_vertical=True)

    elif chart_type == "horizontal_bar":
        bar_locations = predict_bar(image_path)
        print(f"Bounding Boxes: {bar_locations}")
        value_pairs = convert_bar_locations(bar_locations, (x_axis_value_per_pixel, x_sample_point), (y_axis_value_per_pixel, y_sample_point), is_vertical=False)
        print(f"Value Pairs: {value_pairs}")
        plot_value_pairs(image_path, value_pairs, is_vertical=False)

    elif chart_type == "dot":
        dot_locations = predict_dot(image_path)
        print(f"Dot Locations: {dot_locations}")
        dot_groups = group_dot_locations(dot_locations)
        analyzed_groups = analyze_dot_groups(dot_groups)
        plot_dot_groups(image_path, analyzed_groups)

    else:
        print("Unknown chart type")

# Example usage:
if __name__ == "__main__":
    image_dir = 'test/images'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_path in image_files:
        chart_prediction(image_path)
