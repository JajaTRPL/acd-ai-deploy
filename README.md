# 🧠 PMLD – YOLOv11 Training & FastAPI Inference Service

Project ini berisi dua komponen utama:

1. **model-ai** → Digunakan untuk training, evaluasi, dan eksperimen dengan YOLOv11 untuk object detection / segmentation.  
2. **fast-api-service** → REST API berbasis FastAPI untuk inference menggunakan model YOLO yang sudah dilatih.

Repo ini bertujuan memberikan pipeline end-to-end mulai dari training model hingga penyajian hasil inference melalui API.

---

## 🚀 Struktur Folder

```
PMLD/
│
├── fast-api-service/        # Service API untuk inference YOLO
│   ├── main.py              # Endpoint FastAPI
│   └── __pycache__/         
│
├── model-ai/
│   ├── datasets/            # Dataset training (ignored)
│   ├── datasets2/           # Dataset tambahan (ignored)
│   ├── runs/                # Output training YOLO (ignored)
│   ├── *.ipynb              # Notebook training & testing YOLOv11
│   ├── *.pt                 # File weight model (ignored)
│   └── bus.jpg              # Contoh input gambar (ignored)
│
├── .gitignore
└── README.md
```

---

## 📌 Fitur Utama

### 🔹 1. Training YOLOv11
Notebook seperti:

- `readyToTrain.ipynb`
- `readyToTrain2.ipynb`
- `readyToTrain3.ipynb`
- `readyToTrain4.ipynb`
- `testing.ipynb`

Digunakan untuk:

- Preprocessing dataset  
- Training YOLOv11 (nano, small, medium, dll.)  
- Evaluasi hasil training  
- Visualisasi metric dan hasil deteksi  

---

### 🔹 2. FastAPI YOLO Inference Service
FastAPI digunakan untuk menyediakan endpoint:

- Upload gambar  
- Menjalankan inference model YOLO  
- Mengembalikan hasil deteksi dalam bentuk JSON  
- (Opsional) Mengembalikan gambar hasil prediksi  

---

## 📦 Instalasi

### 1️⃣ Clone repository

```bash
git clone https://github.com/najwanmuhammad/PMLD.git
cd PMLD
```

### 2️⃣ Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate   # MacOS / Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependency YOLO

Jika menggunakan Ultralytics:

```bash
pip install ultralytics
```

Install dependency FastAPI:

```bash
pip install fastapi uvicorn python-multipart
```

---

## 🏋️ Training Model YOLOv11

Jalankan salah satu notebook:

- `readyToTrain.ipynb`  
- `readyToTrain2.ipynb`  
- `readyToTrain3.ipynb`  
- `readyToTrain4.ipynb`  

Atau menggunakan CLI:

```bash
yolo detect train model=yolov11s.pt data=data.yaml epochs=50 imgsz=640
```

Output training akan otomatis tersimpan di:

```
model-ai/runs/
```

*(Folder ini otomatis di-ignore dari GitHub)*

---

## ⚡ Menjalankan FastAPI untuk Inference

Masuk ke folder:

```bash
cd fast-api-service
```

Jalankan server:

```bash
uvicorn main:app --reload
```

API berjalan di:

- Dokumentasi OpenAPI → http://127.0.0.1:8000/docs  
- Root API → http://127.0.0.1:8000  

---

## 🧪 Contoh Request API

### Endpoint: `/predict`

**Curl Example:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@bus.jpg"
```

**Contoh Response JSON:**

```json
[
  {
    "class": "bus",
    "confidence": 0.92,
    "bbox": [120, 35, 420, 300]
  }
]
```

---

## 📝 `.gitignore` Ringkas

Repository ini sudah meng-ignore:

```
model-ai/datasets/
model-ai/datasets2/
model-ai/runs/
model-ai/*.pt
model-ai/*.jpg
**/__pycache__/
*.ipynb_checkpoints/
*.log
*.tmp
```

---

## 📄 License

MIT License.
PDU

---

## 👤 Kontributor

- **najuju**
- **najuju**
- **najuju**
- **najuju**

---

## 🤝 Penutup

workflow lengkap mulai dari:

**Training YOLOv11 → Evaluasi Model → Deployment Inference via FastAPI.**
