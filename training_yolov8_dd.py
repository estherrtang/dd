from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Load a model
model = YOLO("/home/directwork/dd/runs/detect/train32/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/directwork/anaconda3/datasets/big_drone_dataset/data.yaml", epochs=300, patience=30)  # train the model
Metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/directwork/dd/drones_sky.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
