from fastapi.staticfiles import StaticFiles
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify allowed domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Ensure "static" folder exists
os.makedirs("static", exist_ok=True)

# Mount static files to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Get the current directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative path to model weights
model_path = os.path.join(base_dir, "runs", "detect",
                          "yolov11_m2", "weights", "best.pt")

# Load the YOLO model
model = YOLO(model_path)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = np.array(image)

    # Convert RGBA to RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image)[0]

    tumor_info = []

    for i, box in enumerate(results.boxes.xyxy):  # Iterate over detected boxes
        x_min, y_min, x_max, y_max = map(int, box[:4])

        try:
            # Correct way to extract class ID
            class_id = int(results.boxes.cls[i])
            tumor_type = model.names[class_id]  # Get tumor name from class ID
        except KeyError:
            tumor_type = "Unknown Tumor"  # Handle unexpected class IDs

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, tumor_type, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tumor_info.append({"tumor_type": tumor_type, "coordinates": [
                          x_min, y_min, x_max, y_max]})

    # Save the image and ensure it is written correctly
    output_path = "static/output.jpg"
    success = cv2.imwrite(output_path, image)
    if not success:
        return {"error": "Failed to save output image."}

    response = {
        "image_url": f"http://127.0.0.1:8000/static/output.jpg?t={int(os.path.getmtime(output_path))}",
        "tumor_info": tumor_info
    }
    # Prevent duplicate logging in debug mode
    if os.environ.get("FASTAPI_DEBUG") != "true":
        print(response)
    return response


@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is running. Visit /docs to test."}

# Run with: uvicorn main:app
