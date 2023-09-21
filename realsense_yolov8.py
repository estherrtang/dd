import pyrealsense2 as rs
import cv2
import numpy as np


from ultralytics import YOLO

# Load the YOLOv5 model
# Load your YOLOv5 model here
model = YOLO("runs/detect/train27/weights/best.onnx")

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from the camera
pipeline.start(config)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# Main loop to capture frames from the camera and perform inference
while True:
    # Wait for a new frame from the camera
    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()
    if not frame:
        continue

    # Convert the frame to a numpy array
    img = np.asanyarray(frame.get_data())

    # Perform inference with YOLOv5
    results = model(img)  # YOLOv5 inference

    # Process and display the results
    img = results.imgs[0]  # Get the processed image with annotations
    cv2.imshow('RealSense YOLOv5', img)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cv2.destroyAllWindows()
pipeline.stop()
