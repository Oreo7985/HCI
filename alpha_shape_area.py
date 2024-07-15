import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon, MultiPolygon

def apply_threshold(image, threshold_value=220):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def compute_white_pixel_ratio(binary_image):
    white_pixel_count = np.sum(binary_image == 255)
    total_pixels = binary_image.size
    white_pixel_ratio = white_pixel_count / total_pixels
    return white_pixel_ratio

def compute_alpha_shape_area(binary_image, alpha=1.0):
    points = np.column_stack(np.where(binary_image == 255))
    if points.shape[0] > 3:  # At least 4 points are needed to compute an alpha shape
        alpha_shape = alphashape.alphashape(points, alpha)
        return alpha_shape, alpha_shape.area
    else:
        return None, 0

def compute_centroid(binary_image):
    moments = cv2.moments(binary_image)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return (cX, cY)
    else:
        return (0, 0)

def draw_alpha_shape(image, alpha_shape):
    if alpha_shape is not None:
        if isinstance(alpha_shape, MultiPolygon):
            for polygon in alpha_shape.geoms:  # Properly iterate over geoms in a MultiPolygon
                image = draw_polygon(image, polygon)
        elif isinstance(alpha_shape, Polygon):
            image = draw_polygon(image, alpha_shape)
    return image  # Ensure the modified image is returned

def draw_polygon(image, polygon):
    exterior_coords = np.array(polygon.exterior.coords)
    for i in range(len(exterior_coords) - 1):
        start_point = tuple(exterior_coords[i].astype(int)[::-1])  # Reverse coordinates for OpenCV
        end_point = tuple(exterior_coords[i + 1].astype(int)[::-1])
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    return image  # Ensure the modified image is returned

def analyze_image(image_path, threshold_value=225, alpha=1.0):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("The image could not be loaded. Please check the file path.")

    binary_image = apply_threshold(image, threshold_value)

    white_pixel_ratio = compute_white_pixel_ratio(binary_image)
    alpha_shape, alpha_shape_area = compute_alpha_shape_area(binary_image, alpha)
    centroid = compute_centroid(binary_image)

    analysis = {
        "Feature": ["White Pixel Ratio", "Alpha Shape Area", "Centroid"],
        "Value": [white_pixel_ratio, alpha_shape_area, centroid]
    }

    analysis_df = pd.DataFrame(analysis)

    # Draw alpha shape on the original image
    image_with_alpha_shape = draw_alpha_shape(image.copy(), alpha_shape)

    # Display results only if image_with_alpha_shape is not None
    if image_with_alpha_shape is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image with Alpha Shape')
        plt.imshow(cv2.cvtColor(image_with_alpha_shape, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.title('Binary Image')
        plt.imshow(binary_image, cmap='gray')
        plt.show()
    else:
        print("Failed to generate alpha shape on the image.")

    return analysis_df
# Input Path
image_path = 'video/VR_free_1_tracks.png'
analysis_result = analyze_image(image_path, alpha=0.05)  # Adjust alpha value for shape fitting

print(analysis_result)
