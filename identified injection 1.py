# Import required libraries for image processing
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image for analysis
image_path = '/mnt/data/vein.jpg'

# Read the image
image = cv2.imread(image_path)

# Convert the image to grayscale for processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply contrast enhancement using histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# Use a threshold to detect veins
_, thresh_image = cv2.threshold(equalized_image, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours of veins in the image
contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours to visualize veins
vein_map = image.copy()
cv2.drawContours(vein_map, contours, -1, (0, 255, 0), 2)

# Find the (assumed unsuitable vein) for injection point
def find_unsuitable_vein(contours):
    min_contour = min(contours, key=cv2.contourArea)
    M = cv2.moments(min_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

# Find the unsuitable vein
unsuitable_vein = find_unsuitable_vein(contours)

# Highlight the unsuitable vein in the image
if unsuitable_vein:
    cv2.circle(vein_map, unsuitable_vein, 8, (0, 0, 255), -1)

# Display the processed result
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(vein_map, cv2.COLOR_BGR2RGB))
plt.title('Processed Image (Unsuitable Vein Highlighted)')
plt.axis('off')
plt.show()

# Return the detected unsuitable vein coordinates
unsuitable_vein
