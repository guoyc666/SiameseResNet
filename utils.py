import os
import torch
from PIL import Image
from torchvision import transforms
import cv2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image(path):
    # image = Image.open(path)
    image_cv2 = cv2.imread(path)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv2)
    image = transform(image_pil)
    return image

def load_support(folder):
    images = []
    for label in os.listdir(folder):
        class_folder = os.path.join(folder, label)
        if not os.path.isdir(class_folder):
            continue
        for filename in os.listdir(class_folder):
            if not filename.endswith('.png'):
                continue
            filepath = os.path.join(class_folder, filename)
            image = get_image(filepath)
            images.append((image, label))
    return images

def load_query(folder):
    images = []
    for filename in os.listdir(folder):
        if not filename.endswith('.png'):
            continue
        filepath = os.path.join(folder, filename)
        image = get_image(filepath)
        images.append((image, filename))
    return images