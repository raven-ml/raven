import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
img = cv2.imread(image_path)
# Convert BGR (OpenCV default) to RGB for proper display in matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
# plt.title("Image")
# plt.savefig("output.png")
# plt.close()
plt.show()