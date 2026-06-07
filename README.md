# Dew-Based IoT Disaster System

A privacy-preserving **real-time disaster detection** system using an Ensemble CNN. Combines Dew (IoT Mobile Devices), Fog (Local Servers), and Cloud (AWS) computing layers to enable offline operation, fast alerts with live GPS tracking, automatic cloud sync, and decentralized emergency response in low-connectivity environments.

---

## 🌟 The Core Idea
The core idea of the project is an end-to-end disaster response pipeline designed for resilience:
1. **IoT / Edge Devices** (like drones and mobile phones) capture images of a disaster along with GPS coordinates (longitude/latitude).
2. The data is sent to the **Dew Layer** (a local server) which instantly predicts the type of disaster and sends alerts to authorities **without the need for the internet**.
3. It gathers information from all local devices and sends just the mathematical "weights" up to the **Federated Learning Layer** to train the global AI and increase accuracy, protecting privacy and saving bandwidth.
4. Finally, when internet is restored, it stores all results, logs, images, locations, and disaster types into **AWS Cloud Services** for global monitoring and record keeping.

---

## Architecture

```
Cloud Layer    ->  AWS DynamoDB + S3
Fog Layer      ->  iot_api.py (FastAPI port 8000) & streamlit_dashboard.py (port 8501)
Dew Layer      ->  IoT Mobile Devices (connecting via browser to port 8000/mobile)
```

- **IoT Mobile Devices (Dew)** act as field nodes, capturing images and GPS location. Images are passed to the Fog server for inference.
- **Fog Server (IoT API)** receives the images, runs inference using a local Ensemble model (EfficientNet-B4 + ResNet50), and logs the prediction (plus GPS info) to a local SQLite database.
- **Dashboard** provides real-time monitoring of alerts via an interactive map, and automatically syncs the offline database to the Cloud when an internet connection is available.

---

## Installation

```bash
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart qrcode[pil] pydeck
```

---

## Dataset & Model Setup

Organize your disaster images as an `ImageFolder` structure inside a `data/` folder:

```
data/
+-- Drought/
+-- Earthquake/
+-- Human_Damage/
+-- Infrastructure/
+-- Land_Slide/
+-- Urban_Fire/
+-- Water_Disaster/
+-- Wild_Fire/
+-- Non_Damage/
```

> **Tip:** Run `python backend/setup_and_train.py` to auto-download datasets and train the standalone global model before running the application.

---

## Running the System

> **All commands should be run from the project root directory.**

### 1. Start the Fog IoT Server
```bash
python backend/iot_api.py
```
This starts the FastAPI server on port 8000. It hosts the API for IoT devices and the `/mobile` web page.

### 2. Connect Field Devices (Dew Layer)
- Ensure your mobile device is on the same local network as the server.
- Navigate to `http://<YOUR_LOCAL_IP>:8000/mobile` on your mobile browser (or scan the QR code in the Dashboard).
- Use the web interface to take a photo. It will gather your GPS location and send the image for immediate processing.

### 3. Launch the Dashboard
```bash
streamlit run frontend/streamlit_dashboard.py
```
View live alerts on a map, manually upload images, check system history, and push data to AWS.

---

## Model

| Component | Detail |
|-----------|--------|
| Architecture | **Ensemble: EfficientNet-B4 + ResNet50** (soft voting) |
| Input size | 224 x 224 RGB |
| Classes | 9 disaster categories |
| Inference | Run locally on Fog server (milliseconds latency) |

---

## Offline Operation & Auto-Cloud Sync

The system is designed to work **fully offline**:

- When internet is unavailable, disaster alerts and captured images are queued in the local SQLite database (`data/cloud_database.sqlite`).
- When connectivity is restored, use the **Dashboard**'s Sync feature or toggle **Online Mode** to automatically push all buffered alerts and images to **AWS S3** and **DynamoDB**.

---

## Project Structure

```
project/
├── frontend/                    ← UI layer
│   └── streamlit_dashboard.py   ← Main monitoring dashboard (with Map)
│
├── backend/                     ← ML & API layer
│   ├── iot_api.py               ← FastAPI server for mobile IoT uploads
│   ├── train_model.py           ← EfficientNet+ResNet50 Ensemble model
│   ├── aws_manager.py           ← AWS S3 + DynamoDB integration
│   ├── setup_and_train.py       ← Download datasets + standalone training
│   ├── resume_train.py          ← Resume from checkpoint
│   └── eval_final.py            ← Final model evaluation
│
├── database/                    ← Data persistence layer
│   └── db_manager.py            ← SQLite offline buffer + cloud sync
│
├── utils/                       ← Shared utilities
├── scripts/                     ← Automation scripts
├── data/                        ← Disaster image classes + SQLite DB
├── models/                      ← Saved model checkpoints (.pth)
└── requirements.txt
```

---

*Dew-Based IoT Disaster System • Research Prototype (c) 2026*
