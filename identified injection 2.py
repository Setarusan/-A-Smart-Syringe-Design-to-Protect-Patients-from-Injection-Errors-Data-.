# Import required libraries for image processing
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path, unsuitable=False):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement using histogram equalization or CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # Use a threshold to detect veins
    _, thresh_image = cv2.threshold(enhanced_image, 50, 255, cv2.THRESH_BINARY if not unsuitable else cv2.THRESH_BINARY_INV)

    # Find contours of veins in the image
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours to visualize veins
    vein_map = image.copy()
    cv2.drawContours(vein_map, contours, -1, (0, 255, 0), 2)

    if unsuitable:
        # Find the smallest contour (assumed unsuitable vein)
        min_contour = min(contours, key=cv2.contourArea)
        M = cv2.moments(min_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            unsuitable_vein = (cx, cy)
            # Highlight the unsuitable vein in the image
            cv2.circle(vein_map, unsuitable_vein, 8, (0, 0, 255), -1)
            return vein_map, unsuitable_vein
    else:
        # Find the largest contour (assumed suitable vein for injection)
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            suitable_vein = (cx, cy)
            # Highlight the suitable vein in the image
            cv2.circle(vein_map, suitable_vein, 10, (0, 255, 0), -1)
            return vein_map, suitable_vein

    return vein_map, None

# Process the first image for unsuitable vein
time_path = '/mnt/data/vein.jpg'
vein_map_unsuitable, unsuitable_vein = process_image(image_path, unsuitable=True)

# Process the second image for suitable vein
image_path_2 = '/mnt/data/PHOTO-2024-10-24-18-57-39 2.jpeg'
vein_map_suitable, suitable_vein = process_image(image_path_2, unsuitable=False)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(vein_map_unsuitable, cv2.COLOR_BGR2RGB))
plt.title("Unsuitable Vein Highlighted")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(vein_map_suitable, cv2.COLOR_BGR2RGB))
plt.title("Suitable Vein Highlighted")
plt.axis('off')

plt.show()

# Output the coordinates of unsuitable and suitable veins
print("Unsuitable Vein Coordinates:", unsuitable_vein)
print("Suitable Vein Coordinates:", suitable_vein)

