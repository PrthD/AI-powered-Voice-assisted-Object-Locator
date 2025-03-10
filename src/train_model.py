import torch
from ultralytics import YOLO

#Create the model from the .pt file
model = YOLO("../models/yolov8s.pt")

#Train the model on the object 365 data
results = model.train(data="../models/Objects365.yaml", epochs=3)

#Save the model
torch.save(model.state_dict(), "../models/yolov8s.pt")