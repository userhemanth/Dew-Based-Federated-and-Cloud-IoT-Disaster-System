# app.py  —  Hugging Face Spaces Entry Point
# Dew-Based Federated and Cloud IoT Disaster System
# A single self-contained Streamlit app (no separate FastAPI backend needed)

import sys, os
import io
import streamlit as st
import pandas as pd
import time
import uuid
from datetime import datetime
from PIL import Image as PILImage

# ── Path Setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "database"))
sys.path.insert(0, os.path.join(_ROOT, "backend"))

# ── Optional Imports ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from db_manager import (
        buffer_local_alert, get_all_buffered,
        get_pending_alerts, sync_to_cloud, cloud_stats
    )
    HAS_DB = True
except ImportError:
    HAS_DB = False

try:
    import aws_manager
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

try:
    import pydeck as pdk
    HAS_PYDECK = True
except ImportError:
    HAS_PYDECK = False

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dew-IoT Disaster System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1117 50%, #0a0f1e 100%); color: #e6edf3; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); border-right: 1px solid #21262d; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border: 1px solid #21262d; border-radius: 14px; padding: 18px 22px; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.02); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 9px !important; color: #8b949e !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1f6feb, #388bfd) !important; color: #fff !important; }
.alert-card { background: rgba(188,36,36,0.10); border-left: 4px solid #f85149; border-radius: 10px; padding: 10px 16px; margin: 5px 0; color: #ffa198; }
.safe-card  { background: rgba(35,134,54,0.10); border-left: 4px solid #3fb950; border-radius: 10px; padding: 10px 16px; margin: 5px 0; color: #7ee787; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.73rem; font-weight: 700; text-transform: uppercase; }
.badge-green  { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; }
.badge-orange { background: rgba(210,153,34,0.15); color: #d2a21f; border: 1px solid #d2a21f; }
.badge-red    { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f85149; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DISASTER_CLASSES = ["Drought", "Earthquake", "Land_Slide", "Non_Damage", "Water_Disaster", "Wild_Fire"]
CLASS_COLORS = {
    "Drought": "#d2a8ff", "Earthquake": "#ff7b72", "Land_Slide": "#a5d6ff",
    "Water_Disaster": "#58a6ff", "Wild_Fire": "#d29922", "Non_Damage": "#3fb950"
}
CLASS_EMOJI = {
    "Drought": "🌵", "Earthquake": "🌍", "Land_Slide": "⛰️",
    "Water_Disaster": "🌊", "Wild_Fire": "🔥", "Non_Damage": "✅"
}
MODEL_PATH = os.path.join(_ROOT, "models", "global_model.pth")

# ── Session State ─────────────────────────────────────────────────────────────
if "online_mode" not in st.session_state:
    st.session_state.online_mode = False
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# ── Model Loading ─────────────────────────────────────────────────────────────
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]) if HAS_TORCH else None

@st.cache_resource(show_spinner="🧠 Loading AI model... (this takes ~30 seconds)")
def load_model():
    if not HAS_TORCH:
        return None, "PyTorch not installed"
    try:
        from train_model import DisasterEnsemble
        model = DisasterEnsemble(num_classes=len(DISASTER_CLASSES), pretrained=False)
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            model.eval()
            return model, "✅ Loaded trained weights from global_model.pth"
        else:
            # No model file found — return None with a clear message
            return None, "⚠️ No model file found at models/global_model.pth. Please upload your trained model."
    except Exception as e:
        return None, f"❌ Error loading model: {e}"

def run_inference(pil_image) -> dict:
    model, msg = load_model()
    if model is None:
        return {"error": msg}
    tensor = EVAL_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    idx = torch.argmax(probs).item()
    return {
        "label": DISASTER_CLASSES[idx],
        "confidence": probs[idx].item(),
        "scores": {DISASTER_CLASSES[i]: float(probs[i]) for i in range(len(DISASTER_CLASSES))}
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌤️ Dew-IoT System")
    st.markdown("**Three-Layer Disaster Network**")
    st.markdown("---")

    # Model status
    st.markdown("### 🧠 AI Model Status")
    if HAS_TORCH:
        if os.path.exists(MODEL_PATH):
            st.markdown('<span class="badge badge-green">✅ MODEL READY</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-red">❌ MODEL FILE MISSING</span>', unsafe_allow_html=True)
            st.caption("Upload `global_model.pth` to the `models/` folder in your Space.")
    else:
        st.markdown('<span class="badge badge-red">❌ PYTORCH NOT FOUND</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌐 Network Mode")
    st.session_state.online_mode = st.toggle("Online / AWS Sync Mode", value=st.session_state.online_mode)
    if st.session_state.online_mode:
        st.markdown('<span class="badge badge-green">● ONLINE</span>', unsafe_allow_html=True)
        if HAS_DB:
            pending = get_pending_alerts()
            if pending and st.button(f"☁️ Sync {len(pending)} pending alert(s) to AWS"):
                sync_to_cloud()
                st.rerun()
    else:
        st.markdown('<span class="badge badge-orange">◎ OFFLINE</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh"):
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌤️ Dew-Based IoT Disaster System")
st.markdown("*Privacy-preserving disaster detection using Federated Learning & Ensemble CNN*")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖼️ Upload & Detect", "📊 Statistics", "⚠️ Alert History", "☁️ Cloud Database", "🌐 Architecture"
])

# ── TAB 1: UPLOAD & DETECT ────────────────────────────────────────────────────
with tab1:
    st.subheader("🖼️ Disaster Image Detection")
    st.markdown("Upload an image from a drone, phone, or satellite to run AI analysis.")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "webp"],
            help="Supports JPG, PNG, and WEBP formats"
        )
        device_name = st.text_input("Device/Source Name", value="Web Upload", placeholder="e.g. Drone #1, Satellite Feed")
        lat = st.number_input("Latitude (optional)", value=0.0, format="%.6f")
        lon = st.number_input("Longitude (optional)", value=0.0, format="%.6f")

        if uploaded:
            pil_image = PILImage.open(uploaded).convert("RGB")
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if uploaded:
            if st.button("🔍 Run Disaster Detection", use_container_width=True, type="primary"):
                with st.spinner("Analyzing image with AI..."):
                    result = run_inference(pil_image)

                if "error" in result:
                    st.error(result["error"])
                else:
                    label     = result["label"]
                    conf      = result["confidence"]
                    color     = CLASS_COLORS.get(label, "#ffffff")
                    emoji     = CLASS_EMOJI.get(label, "❓")

                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.04); border:1px solid #30363d;
                                border-radius:16px; padding:24px; margin-bottom:16px; text-align:center;'>
                        <div style='font-size:3rem;'>{emoji}</div>
                        <h2 style='color:{color}; margin:8px 0;'>{label}</h2>
                        <div style='font-size:2rem; font-weight:700; color:#e6edf3;'>{conf*100:.1f}%</div>
                        <div style='color:#8b949e; font-size:0.85rem;'>Confidence Score</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability bars for all classes
                    st.markdown("#### 📊 All Class Probabilities")
                    for cls, score in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True):
                        c = CLASS_COLORS.get(cls, "#8b949e")
                        st.markdown(f"**{CLASS_EMOJI.get(cls,'')} {cls}**")
                        st.progress(score, text=f"{score*100:.1f}%")

                    # Buffer to local DB
                    if HAS_DB:
                        buffer_local_alert(
                            device_id=str(uuid.uuid4())[:8],
                            device_name=device_name,
                            label=label,
                            confidence=conf,
                            latitude=lat if lat != 0.0 else None,
                            longitude=lon if lon != 0.0 else None
                        )
                        st.success("✅ Alert buffered to local database.")

                    # Save to session state for statistics
                    st.session_state.predictions.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "device": device_name,
                        "label": label,
                        "confidence": f"{conf*100:.1f}%"
                    })
        else:
            st.info("👆 Upload an image on the left to start analysis.")
            st.markdown("""
            ### How it works:
            1. **Upload** a disaster image (photo from drone, phone, or satellite)
            2. **Click** Run Disaster Detection
            3. **AI analyzes** the image using 3 neural networks (EfficientNet + ConvNeXt + ViT)
            4. **Alert is buffered** locally for offline resilience
            5. **When online**, alerts sync to AWS Cloud automatically
            """)

# ── TAB 2: STATISTICS ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Session Statistics")
    if st.session_state.predictions:
        df = pd.DataFrame(st.session_state.predictions)
        total  = len(df)
        disasters = df[df["label"] != "Non_Damage"]
        safe      = df[df["label"] == "Non_Damage"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Analyzed", total)
        m2.metric("🚨 Disasters Detected", len(disasters))
        m3.metric("✅ Safe Zones", len(safe))

        st.markdown("#### Recent Predictions")
        st.dataframe(df[::-1], use_container_width=True)

        st.markdown("#### Disaster Breakdown")
        breakdown = df["label"].value_counts().reset_index()
        breakdown.columns = ["Disaster Type", "Count"]
        st.bar_chart(breakdown.set_index("Disaster Type"))
    else:
        st.info("No predictions yet. Go to **Upload & Detect** tab to analyze an image.")

# ── TAB 3: ALERT HISTORY ──────────────────────────────────────────────────────
with tab3:
    st.subheader("⚠️ Alert History (Local Buffer)")
    if HAS_DB:
        alerts = get_all_buffered()
        if alerts:
            df = pd.DataFrame(alerts)
            cols_to_show = [c for c in ["timestamp", "device_name", "label", "confidence", "latitude", "longitude", "synced"] if c in df.columns]
            df_show = df[cols_to_show].copy()
            if "synced" in df_show.columns:
                df_show["synced"] = df_show["synced"].map({1: "✅ Synced", 0: "⏳ Pending"})
            if "confidence" in df_show.columns:
                df_show["confidence"] = df_show["confidence"].apply(lambda x: f"{float(x)*100:.1f}%" if x else "N/A")
            st.dataframe(df_show, use_container_width=True)

            # Map view
            if HAS_PYDECK:
                map_data = [
                    {"lat": a["latitude"], "lon": a["longitude"],
                     "label": a["label"], "device": a.get("device_name", ""),
                     "color": [63, 185, 80] if a["label"] == "Non_Damage" else [248, 81, 73]}
                    for a in alerts if a.get("latitude") and a.get("longitude")
                ]
                if map_data:
                    st.markdown("#### 🗺️ GPS Map")
                    df_map = pd.DataFrame(map_data)
                    layer = pdk.Layer("ScatterplotLayer", df_map,
                                      get_position='[lon, lat]', get_color='color',
                                      get_radius=500, pickable=True)
                    view = pdk.ViewState(latitude=df_map['lat'].mean(),
                                        longitude=df_map['lon'].mean(), zoom=8)
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                             tooltip={"text": "{device}\n{label}"}))
        else:
            st.info("No alerts buffered yet.")
    else:
        st.warning("Database module not available.")

# ── TAB 4: CLOUD DB ───────────────────────────────────────────────────────────
with tab4:
    st.subheader("☁️ Cloud Database (AWS DynamoDB + S3)")
    aws_ok = HAS_AWS and aws_manager.is_aws_configured()

    if aws_ok:
        st.markdown('<span class="badge badge-green">✅ AWS CONNECTED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-orange">❌ AWS NOT CONFIGURED</span>', unsafe_allow_html=True)
        st.info("To enable cloud sync, set your AWS credentials as **Secrets** in your Hugging Face Space settings:\n\n`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`")

    if HAS_DB:
        stats = cloud_stats()
        c1, c2 = st.columns(2)
        c1.metric("Total Buffered Locally", stats.get("total_cloud_alerts", 0))
        c2.metric("Pending Sync to Cloud", stats.get("pending_sync", 0))

    if aws_ok and st.session_state.online_mode:
        recs = aws_manager.fetch_aws_alerts()
        if recs:
            st.markdown("### 📋 DynamoDB Records")
            df_aws = pd.DataFrame(recs)
            st.dataframe(df_aws, use_container_width=True)

# ── TAB 5: ARCHITECTURE ───────────────────────────────────────────────────────
with tab5:
    st.subheader("🌐 Three-Layer Architecture")

    st.markdown("""
    This system implements a **three-layer hierarchical architecture** for disaster detection
    that works even when the internet is completely down.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ### 📱 Dew Layer
        **IoT Field Devices**
        - Mobile phones & drones
        - Capture images in real-time
        - Send to Fog Layer via local Wi-Fi
        - Works completely offline
        """)
    with c2:
        st.markdown("""
        ### 🌫️ Fog Layer
        **Local AI Server (This App)**
        - Runs Ensemble CNN (EfficientNet + ConvNeXt + ViT)
        - Makes predictions without internet
        - Buffers alerts in SQLite locally
        - Performs Federated Learning aggregation
        """)
    with c3:
        st.markdown("""
        ### ☁️ Cloud Layer
        **AWS DynamoDB + S3**
        - Receives synced alerts when online
        - Stores images permanently in S3
        - Enables global monitoring
        - Coordinates rescue operations
        """)

    st.markdown("---")
    st.code("""
┌────────────────────────────────────────────────────────────┐
│  ☁️  CLOUD LAYER  (AWS DynamoDB + S3)                      │
│  Permanent storage — synced only when internet available   │
└───────────────────────▲────────────────────────────────────┘
                        │  Auto-sync via Internet
┌───────────────────────▼────────────────────────────────────┐
│  🌫️  FOG LAYER  (Dew Aggregator — This Streamlit App)     │
│  Local AI inference, SQLite buffer, Federated Aggregation  │
└───────────────────────▲────────────────────────────────────┘
                        │  POST image (local Wi-Fi)
┌───────────────────────▼────────────────────────────────────┐
│  📱  DEW LAYER  (IoT Mobile / Drone Devices)               │
│  Field devices capturing images + GPS in real-time         │
└────────────────────────────────────────────────────────────┘
    """)
