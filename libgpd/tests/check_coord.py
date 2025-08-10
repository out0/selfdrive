import cv2
import numpy as np

# Load the image
img = cv2.imread("debug.png")

# Read coordinates from coord.txt
with open('coords.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        x_str, y_str = line.split(',')
        x, y = int(x_str.strip()), int(y_str.strip())
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x] = (255, 0, 0)  # Blue in BGR

# Save the modified image
cv2.imwrite('debug2.png', img)