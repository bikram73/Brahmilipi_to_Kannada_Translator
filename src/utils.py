import os
import cv2

def save_character_images(image, text):
    os.makedirs("images", exist_ok=True)
    for char in text:
        char_path = os.path.join("images", f"{char}.jpg")
        cv2.imwrite(char_path, image)
        print(f"Saved: {char_path}")

def ensure_directories():
    required_dirs = ['images', 'data', 'src', 'output']
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")