import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the second image
image_path_2 = '/mnt/data/PHOTO-2024-10-24-18-57-39 2.jpeg'
image_2 = cv2.imread(image_path_2)

# Convert to grayscale for processing
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# Apply a contrast-limited adaptive histogram equalization to enhance veins
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_2 = clahe.apply(gray_2)

# Thresholding to extract prominent veins
_, thresh_2 = cv2.threshold(enhanced_2, 50, 255, cv2.THRESH_BINARY)

# Detect contours in the thresholded image
contours, _ = cv2.findContours(thresh_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and mark an approximate center of vein cluster
output_image_2 = image_2.copy()

# Find largest contour (likely main vein area)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        # Draw a green point at the injection site
        cv2.circle(output_image_2, (center_x, center_y), 10, (0, 255, 0), -1)

# Display result
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output_image_2, cv2.COLOR_BGR2RGB))
plt.title("Identified Injection Site in Image 2")
plt.axis('off')
plt.show()

(center_x, center_y)
