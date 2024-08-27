from ultralytics import YOLO

# load trained model
model = YOLO('models/best.pt')

# display model information (optional)
model.info()

# run inference with the trained model
results = model.predict(source="input_videos/08fd33_4.mp4", save=True)

print(results[0])