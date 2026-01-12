from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io

# ===========================
# 1. Init FastAPI + CORS
# ===========================
app = FastAPI(
    title="YOLOv11 Segmentation Service",
    description="Automated Cutting Description - YOLOv11 API",
    version="1.0.0",
)

# kalau nanti mau diakses langsung dari Next.js origin, isi origin-nya di sini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti ke ["http://localhost:3000"] kalau mau lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# 2. Load model sekali di awal
# ===========================
# pakai model hasil trainingmu
MODEL_PATH = "../model-ai/fixModel.pt"
model = YOLO(MODEL_PATH)

# threshold hasil sweep tadi
CONF_TH = 0.55
IOU_TH = 0.45
IMG_SIZE = 640

# ===========================
# 3. Response schema
# ===========================
class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

class PredictionResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection]


# ===========================
# 4. Routes
# ===========================
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- baca file -> PIL image ---
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # --- inference YOLO ---
    results = model(
        image,
        imgsz=IMG_SIZE,
        conf=CONF_TH,
        iou=IOU_TH,
        verbose=False,
    )[0]  # ambil hasil pertama

    detections: List[Detection] = []

    if results.boxes is not None:
        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes_xyxy, classes, confs):
            x1, y1, x2, y2 = box.tolist()
            cls_id_int = int(cls_id)
            cls_name = results.names[cls_id_int]

            detections.append(
                Detection(
                    class_name=cls_name,
                    class_id=cls_id_int,
                    confidence=float(conf),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

    # =====================================
    # ➕ Tambahkan summary perhitungan class
    # =====================================
    sandstone_count = sum(1 for d in detections if d.class_name == "sandstone")
    siltstone_count = sum(1 for d in detections if d.class_name == "siltstone")
    total = len(detections)

    summary = {
        "sandstone_count": sandstone_count,
        "siltstone_count": siltstone_count,
        "total_instances": total,
        "sandstone_percentage": (sandstone_count / total * 100) if total > 0 else 0,
        "siltstone_percentage": (siltstone_count / total * 100) if total > 0 else 0,
    }

    # =====================================
    # Return dict biasa (bukan PredictionResponse)
    # =====================================
    return {
        "width": width,
        "height": height,
        "detections": [d.dict() for d in detections],
        "summary": summary,
    }

    
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    # === 1. Baca gambar ===
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # === 2. YOLO predict ===
    results = model(
        image,
        imgsz=IMG_SIZE,
        conf=CONF_TH,
        iou=IOU_TH,
        verbose=False,
    )[0]

    # === 3. Render hasil (mask + bbox + class label) ===
    rendered = results.plot()   # numpy array BGR

    # === 4. Convert numpy → bytes (JPEG) ===
    _, buffer = cv2.imencode(".jpg", rendered)
    img_bytes = io.BytesIO(buffer.tobytes())

    # === 5. Return sebagai image stream ===
    return StreamingResponse(
        img_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=predict.jpg"}
    )
