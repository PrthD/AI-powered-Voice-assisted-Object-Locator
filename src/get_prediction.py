import torch
from ultralytics import YOLO
import cv2
import Cam
import numpy as np

def predict(model):
    return_dict = Cam.sample()
    #3D image array
    image = return_dict["frame"]
    # Resize the image to the required input size
    image_resized = cv2.resize(image, (640, 640))

    # Convert the image to RGB (if it's BGR)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normalize the image
    image_normalized = image_rgb / 255.0

    # Convert to a 4D tensor: (batch_size, height, width, channels) -> (1, 640, 640, 3)
    image_tensor = np.expand_dims(image_normalized, axis=0)

    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(image_tensor).float()

    # You may need to permute the dimensions to match PyTorch's expected input format (channels first)
    image_tensor = image_tensor.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)

    # Make the prediction (forward pass)
    results = model(image_tensor)  # The input should be a batch (4D tensor)

    return results.xywh[0] # Format: [x_center, y_center, width, height, confidence, class_id]
