# backend/iot_api.py
import sys, os
import io
import time
import uuid
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage
from torchvision import transforms

# Fix imports
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_ROOT, "database"))
sys.path.insert(0, os.path.join(_ROOT, "utils"))
sys.path.insert(0, os.path.dirname(__file__))

from db_manager import buffer_local_alert, get_cloud_alerts, get_all_buffered, get_pending_alerts, mark_alert_synced_to_aws
from train_model import DisasterEnsemble
from aws_manager import is_aws_configured, push_alert_to_aws
import asyncio
from contextlib import asynccontextmanager

# -----------------------------------------------------------------------
# AWS BACKGROUND SYNC
# -----------------------------------------------------------------------

async def aws_sync_worker():
    print("[AWS Sync] Background worker started. Checking for pending alerts every 60s...")
    while True:
        if is_aws_configured():
            pending = get_pending_alerts()
            for alert in pending:
                image_bytes = None
                img_path = alert.get("image_path")
                if img_path and os.path.exists(img_path):
                    try:
                        with open(img_path, "rb") as f:
                            image_bytes = f.read()
                    except Exception as e:
                        print(f"[AWS Sync] Error reading image {img_path}: {e}")
                
                res = push_alert_to_aws(
                    device_id=alert["device_id"],
                    device_name=alert.get("device_name") or "Unknown Device",
                    label=alert["label"],
                    confidence=alert["confidence"],
                    timestamp=alert["timestamp"],
                    latitude=alert.get("latitude") or 0.0,
                    longitude=alert.get("longitude") or 0.0,
                    gps_accuracy=alert.get("gps_accuracy") or 0.0,
                    image_bytes=image_bytes
                )
                
                if res.get("success"):
                    print(f"[AWS Sync] Successfully synced alert {alert['id']} to AWS.")
                    mark_alert_synced_to_aws(alert["id"], res.get("image_url", ""))
                else:
                    print(f"[AWS Sync] Failed to sync alert {alert['id']}: {res.get('error')}")
        
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan_context(app: FastAPI):
    asyncio.create_task(aws_sync_worker())
    yield

# -----------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------
app = FastAPI(title="Dew IoT Fog Server", lifespan=lifespan_context)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join(_ROOT, "models", "global_model.pth")
CLASS_NAMES = [
    "Drought", "Earthquake",
    "Land_Slide", "Non_Damage", "Water_Disaster", "Wild_Fire"
]

# Initialize model
model = DisasterEnsemble(num_classes=len(CLASS_NAMES), pretrained=False)
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded trained model from {MODEL_PATH}")
else:
    print("[WARN] Model checkpoint not found! Using untrained weights.")

model.to(DEVICE)
model.eval()

# Apply INT8 Dynamic Quantization for faster CPU inference
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("[INFO] Model dynamically quantized to INT8.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])



# -----------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    device_id: str = Form(...),
    device_name: str = Form("Unknown Device"),
    latitude: float = Form(None),
    longitude: float = Form(None),
    gps_accuracy: float = Form(None),
    captured_at: str = Form(None)
):
    """Receive image from IoT device, run inference, save to DB."""
    try:
        contents = await image.read()
        pil_img = PILImage.open(io.BytesIO(contents)).convert("RGB")
        

        # Inference
        img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img_t)
            probs = torch.nn.functional.softmax(out[0], dim=0)
            conf, pred_idx = torch.max(probs, 0)
        
        label = CLASS_NAMES[pred_idx.item()]
        confidence = conf.item()

        # Save image locally in label-specific folder for Federated Learning
        alerts_dir = os.path.join(_ROOT, "data", "offline_images", label)
        os.makedirs(alerts_dir, exist_ok=True)
        filename = f"{device_id}_{int(time.time())}.jpg"
        img_path = os.path.join(alerts_dir, filename)
        pil_img.save(img_path)
        
        scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        if not captured_at:
            captured_at = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save to database
        alert_id = buffer_local_alert(
            device_id=device_id,
            device_name=device_name,
            label=label,
            confidence=confidence,
            latitude=latitude,
            longitude=longitude,
            gps_accuracy=gps_accuracy,
            captured_at=captured_at,
            image_path=img_path
        )

        # -------------------------------------------------------------
        # SIMULATE OFFLINE EMERGENCY BROADCAST
        # -------------------------------------------------------------
        is_disaster = (label != "Non_Damage")
        if is_disaster and confidence > 0.50:
            print("\n" + "="*60)
            print("🚨 OFFLINE EMERGENCY BROADCAST TRIGGERED 🚨")
            print(f"   TYPE: {label} ({confidence*100:.1f}%)")
            print(f"   LOC: Lat {latitude}, Lon {longitude}")
            print(f"   Routing alert to local authorities via LoRa/SMS simulation...")
            print("="*60 + "\n")
            
            log_file = os.path.join(_ROOT, "data", "local_authorities_log.txt")
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{captured_at}] EMERGENCY: {label} detected at {latitude}, {longitude} by {device_name}.\n")
        # -------------------------------------------------------------

        return JSONResponse(content={
            "alert_id": alert_id,
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "is_disaster": label != "Non_Damage",
            "device_id": device_id,
            "device_name": device_name,
            "latitude": latitude,
            "longitude": longitude,
            "captured_at": captured_at,
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/alerts/live")
def get_live_alerts():
    """Return latest buffered alerts for the dashboard."""
    alerts = get_all_buffered()
    return {"alerts": alerts[:50]}  # Return top 50 recent


# -----------------------------------------------------------------------
# MOBILE HTML PAGE
# -----------------------------------------------------------------------

@app.get("/mobile", response_class=HTMLResponse)
def serve_mobile_page():
    """Serves a mobile-optimized HTML page for capturing images with GPS."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Dew IoT Field Device</title>
        <style>
            :root { --bg: #121212; --card: #1E1E1E; --primary: #BB86FC; --text: #E0E0E0; --green: #03DAC6; --red: #CF6679; }
            body { font-family: -apple-system, system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
            .card { background: var(--card); border-radius: 16px; padding: 20px; width: 100%; max-width: 400px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; box-sizing: border-box; }
            h2 { margin-top: 0; font-size: 1.5rem; text-align: center; }
            .status { font-size: 0.9rem; margin-bottom: 15px; padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.05); }
            .status span { font-weight: bold; }
            .gps-on { color: var(--green); }
            .gps-off { color: var(--red); }
            
            #cameraInput { display: none; }
            .btn { display: block; width: 100%; padding: 15px; border-radius: 12px; border: none; font-size: 1.1rem; font-weight: bold; text-align: center; cursor: pointer; transition: 0.2s; box-sizing: border-box; margin-bottom: 10px; }
            .btn-camera { background: #3700B3; color: white; }
            .btn-submit { background: var(--green); color: black; display: none; }
            .btn:active { transform: scale(0.98); }
            
            #preview { width: 100%; border-radius: 8px; margin-top: 15px; display: none; max-height: 300px; object-fit: cover; }
            
            #result { display: none; margin-top: 20px; text-align: center; }
            #result-label { font-size: 1.8rem; font-weight: bold; color: var(--primary); margin: 10px 0; }
            .bar-bg { width: 100%; background: #333; height: 10px; border-radius: 5px; overflow: hidden; }
            .bar-fill { height: 100%; background: var(--green); width: 0%; transition: width 0.5s; }
            
            #loader { display: none; text-align: center; margin-top: 20px; }
            .spinner { width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.1); border-left-color: var(--primary); border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 10px; }
            @keyframes spin { 100% { transform: rotate(360deg); } }
            
            input[type="text"] { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #444; background: #222; color: white; margin-bottom: 15px; box-sizing: border-box; font-size: 1rem; }
        </style>
    </head>
    <body>

    <div class="card">
        <h2>📱 Dew Field Device</h2>
        
        <input type="text" id="deviceName" placeholder="Enter Device Name (e.g. Unit 1)" value="Field Phone 1">
        
        <div class="status">
            GPS: <span id="gpsStatus" class="gps-off">Locating...</span><br>
            <small id="gpsCoords">Lat: -- | Lon: -- | Acc: --</small>
        </div>

        <label class="btn btn-camera" for="cameraInput">📸 Take Photo</label>
        <input type="file" id="cameraInput" accept="image/*" capture="environment">
        
        <img id="preview" alt="Preview">
        
        <button id="submitBtn" class="btn btn-submit">🚀 Send to Dew Server</button>
        
        <div id="loader">
            <div class="spinner"></div>
            <div>Analyzing image locally...</div>
        </div>
        
        <div id="result">
            <h3>Prediction Result</h3>
            <div id="result-label">--</div>
            <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:5px;">
                <span>Confidence</span>
                <span id="result-conf">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill" id="result-bar"></div></div>
            <button class="btn btn-camera" style="margin-top:20px;" onclick="location.reload()">Next Capture</button>
        </div>
    </div>

    <script>
        let currentLat = null;
        let currentLon = null;
        let currentAcc = null;
        let deviceId = localStorage.getItem("dew_device_id");
        
        if (!deviceId) {
            deviceId = "dev_" + Math.random().toString(36).substr(2, 9);
            localStorage.setItem("dew_device_id", deviceId);
        }

        // GPS Tracking
        if (navigator.geolocation) {
            navigator.geolocation.watchPosition(
                (pos) => {
                    currentLat = pos.coords.latitude;
                    currentLon = pos.coords.longitude;
                    currentAcc = pos.coords.accuracy;
                    document.getElementById('gpsStatus').className = 'gps-on';
                    document.getElementById('gpsStatus').innerText = 'Locked';
                    document.getElementById('gpsCoords').innerText = `Lat: ${currentLat.toFixed(5)} | Lon: ${currentLon.toFixed(5)} | Acc: ${Math.round(currentAcc)}m`;
                },
                (err) => {
                    document.getElementById('gpsStatus').innerText = 'Failed';
                    document.getElementById('gpsCoords').innerText = err.message;
                },
                { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
            );
        } else {
            document.getElementById('gpsStatus').innerText = 'Not Supported';
        }

        // Image Preview
        const cameraInput = document.getElementById('cameraInput');
        const preview = document.getElementById('preview');
        const submitBtn = document.getElementById('submitBtn');

        cameraInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
                submitBtn.style.display = 'block';
            }
        });

        // Submit to Server
        submitBtn.addEventListener('click', async () => {
            const file = cameraInput.files[0];
            if (!file) return;

            submitBtn.style.display = 'none';
            document.getElementById('loader').style.display = 'block';

            const formData = new FormData();
            formData.append("image", file);
            formData.append("device_id", deviceId);
            formData.append("device_name", document.getElementById('deviceName').value);
            
            if (currentLat !== null) {
                formData.append("latitude", currentLat);
                formData.append("longitude", currentLon);
                formData.append("gps_accuracy", currentAcc);
            }
            
            const now = new Date();
            // Local ISO string hack
            const offset = now.getTimezoneOffset();
            const localDate = new Date(now.getTime() - (offset*60*1000));
            formData.append("captured_at", localDate.toISOString().slice(0, 19).replace('T', ' '));

            try {
                const res = await fetch("/predict", { method: "POST", body: formData });
                const data = await res.json();
                
                document.getElementById('loader').style.display = 'none';
                document.getElementById('result').style.display = 'block';
                document.getElementById('result-label').innerText = data.label;
                
                const confPercent = Math.round(data.confidence * 100);
                document.getElementById('result-conf').innerText = confPercent + '%';
                document.getElementById('result-bar').style.width = confPercent + '%';
                
                if (data.label === "Non_Damage") {
                    document.getElementById('result-label').style.color = "var(--green)";
                    document.getElementById('result-bar').style.background = "var(--green)";
                } else {
                    document.getElementById('result-label').style.color = "var(--red)";
                    document.getElementById('result-bar').style.background = "var(--red)";
                }
            } catch (err) {
                alert("Upload failed. Are you connected to the Dew server?");
                document.getElementById('loader').style.display = 'none';
                submitBtn.style.display = 'block';
            }
        });
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    print("Starting Dew IoT API Server on port 8000...")
    print("Connect devices to: http://<your-ip>:8000/mobile")
    uvicorn.run("iot_api:app", host="0.0.0.0", port=8000, reload=True)
