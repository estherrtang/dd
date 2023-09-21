from ultralytics import YOLO
import torch

torch.cuda.set_device(0)
# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/directwork/anaconda3/datasets/combined_new_dataset/data.yaml", epochs=300, patience=30)  # train the model
Metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/directwork/dd/drones_sky.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
