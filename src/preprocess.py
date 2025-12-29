import cv2
import numpy as np

def preprocess_image(image):
    if image is None:
        raise ValueError("Input image is None")

    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    return normalized.reshape(64, 64, 1)