# Import required libraries for image processing
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the images for analysis
image1_path = '/mnt/data/WhatsApp_Image_2024-10-21_at_1.16.25_AM__1_-removebg-preview.png'
image2_path = '/mnt/data/download.jpeg'

# Read the images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Convert images to grayscale for processing
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply contrast enhancement using histogram equalization
equalized_image1 = cv2.equalizeHist(gray_image1)
equalized_image2 = cv2.equalizeHist(gray_image2)

# Use a threshold to detect veins
_, thresh_image1 = cv2.threshold(equalized_image1, 50, 255, cv2.THRESH_BINARY_INV)
_, thresh_image2 = cv2.threshold(equalized_image2, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours of veins in both images
contours1, _ = cv2.findContours(thresh_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(thresh_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours to visualize veins
vein_map1 = image1.copy()
vein_map2 = image2.copy()
cv2.drawContours(vein_map1, contours1, -1, (0, 255, 0), 2)
cv2.drawContours(vein_map2, contours2, -1, (0, 255, 0), 2)

# Find the largest contour (assumed vein) for injection point
def find_injection_point(contours):
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

# Find injection points
injection_point1 = find_injection_point(contours1)
injection_point2 = find_injection_point(contours2)

# Highlight the injection points in both images
if injection_point1:
    cv2.circle(vein_map1, injection_point1, 8, (0, 0, 255), -1)
if injection_point2:
    cv2.circle(vein_map2, injection_point2, 8, (0, 0, 255), -1)

# Display the processed results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(vein_map1, cv2.COLOR_BGR2RGB))
axes[0].set_title('Processed Image 1')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(vein_map2, cv2.COLOR_BGR2RGB))
axes[1].set_title('Processed Image 2 (Injection Point Highlighted)')
axes[1].axis('off')

plt.show()

# Return the detected injection point for image 2
injection_point2
