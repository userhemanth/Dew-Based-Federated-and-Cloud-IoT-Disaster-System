# frontend/streamlit_dashboard.py
import sys, os
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_ROOT, "database"))  # db_manager
sys.path.insert(0, os.path.join(_ROOT, "backend"))   # aws_manager, train_model
sys.path.insert(0, _ROOT)                            # project root fallback

import streamlit as st
import pandas as pd
import time
import uuid
import pydeck as pdk
import io
import socket
from datetime import datetime
from PIL import Image as PILImage

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from db_manager import (
        buffer_local_alert, sync_to_cloud,
        get_cloud_alerts, get_pending_alerts, get_all_buffered,
        cloud_stats
    )
    HAS_DB = True
except ImportError:
    HAS_DB = False

try:
    import aws_manager
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dew-IoT Disaster System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1117 50%, #0a0f1e 100%); color: #e6edf3; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); border-right: 1px solid #21262d; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border: 1px solid #21262d; border-radius: 14px; padding: 18px 22px; backdrop-filter: blur(12px); }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.02); border-radius: 12px; padding: 4px; border-bottom: none !important; }
.stTabs [data-baseweb="tab"] { border-radius: 9px !important; color: #8b949e !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1f6feb, #388bfd) !important; color: #ffffff !important; box-shadow: 0 2px 8px rgba(31,111,235,0.35); }
.alert-card { background: rgba(188,36,36,0.10); border-left: 4px solid #f85149; border-radius: 10px; padding: 10px 16px; margin: 5px 0; color: #ffa198; }
.safe-card { background: rgba(35,134,54,0.10); border-left: 4px solid #3fb950; border-radius: 10px; padding: 10px 16px; margin: 5px 0; color: #7ee787; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.73rem; font-weight: 700; text-transform: uppercase; }
.badge-green { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; }
.badge-orange { background: rgba(210,153,34,0.15); color: #d2a21f; border: 1px solid #d2a21f; }
</style>
""", unsafe_allow_html=True)

DISASTER_CLASSES = [
    "Drought", "Earthquake",
    "Land_Slide", "Non_Damage", "Water_Disaster", "Wild_Fire"
]
CLASS_COLORS = {
    "Drought": "#d2a8ff", "Earthquake": "#ff7b72",
    "Land_Slide": "#a5d6ff", "Water_Disaster": "#58a6ff",
    "Wild_Fire": "#d29922", "Non_Damage": "#3fb950"
}
MODEL_PATH = os.path.join(_ROOT, "models", "global_model.pth")

if "online_mode" not in st.session_state:
    st.session_state.online_mode = False

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]) if HAS_TORCH else None

@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    if not HAS_TORCH: return None
    try:
        from train_model import DisasterEnsemble
        model = DisasterEnsemble(num_classes=len(DISASTER_CLASSES), pretrained=False)
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location="cpu")
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
        else:
            model = DisasterEnsemble(num_classes=len(DISASTER_CLASSES), pretrained=True)
        model.eval()
        return model
    except Exception as e:
        return None

def run_inference(pil_image) -> dict:
    model = load_model()
    tensor = EVAL_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    idx = torch.argmax(probs).item()
    return {"label": DISASTER_CLASSES[idx], "confidence": probs[idx].item(), 
            "scores": {DISASTER_CLASSES[i]: float(probs[i]) for i in range(len(DISASTER_CLASSES))}}

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

LOCAL_IP = get_local_ip()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌤️ Dew-IoT System")
    st.markdown("**Three-Layer Disaster Network**")
    st.markdown("---")
    
    st.markdown("### 🌐 Network Connectivity")
    st.session_state.online_mode = st.toggle("Online Mode", value=st.session_state.online_mode)
    
    if st.session_state.online_mode:
        st.markdown('<span class="badge badge-green">● ONLINE</span>', unsafe_allow_html=True)
        if HAS_DB:
            pending = get_pending_alerts()
            if pending and st.button(f"☁️ Sync {len(pending)} alert(s)"):
                sync_to_cloud()
                st.rerun()
    else:
        st.markdown('<span class="badge badge-orange">◎ OFFLINE</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.rerun()

st.markdown("# 🌤️ Dew-Based IoT Disaster System")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖥️ Website Upload", "📱 IoT Device Feed", "⚠️ Alert History", "☁️ Cloud Database", "🌐 Architecture"
])

# ── TAB 1: WEBSITE UPLOAD ────────────────────────────────────────────────────
with tab1:
    st.subheader("🖥️ Manual Website Upload")
    c1, c2 = st.columns(2)
    with c1:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded:
            pil_image = PILImage.open(uploaded).convert("RGB")
            st.image(pil_image, use_container_width=True)
            if st.button("Run Inference", use_container_width=True):
                res = run_inference(pil_image)
                with c2:
                    st.markdown("### Result")
                    color = CLASS_COLORS.get(res["label"], "#fff")
                    st.markdown(f"<h2 style='color:{color}'>{res['label']} ({res['confidence']*100:.1f}%)</h2>", unsafe_allow_html=True)
                    if HAS_DB:
                        buffer_local_alert(
                            device_id="web_upload", device_name="Website Portal",
                            label=res["label"], confidence=res["confidence"],
                            latitude=None, longitude=None
                        )
                        st.success("Buffered to local DB.")

# ── TAB 2: IOT DEVICE FEED ───────────────────────────────────────────────────
with tab2:
    st.subheader("📱 IoT Live Device Feed")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### Connect Mobile Device")
        backend_base = os.environ.get("BACKEND_URL", f"http://{LOCAL_IP}:8000")
        url = f"{backend_base}/mobile"
        st.markdown(f"**URL:** `{url}`")
        try:
            import qrcode
            qr = qrcode.make(url)
            buf = io.BytesIO()
            qr.save(buf, format="PNG")
            st.image(buf.getvalue(), width=250, caption="Scan with Phone Camera")
        except:
            st.warning("Install qrcode to see the scan code.")
            
    with c2:
        st.markdown("#### Live Map")
        if HAS_DB:
            alerts = get_all_buffered()
            map_data = []
            for a in alerts:
                if a.get('latitude') and a.get('longitude'):
                    color = [63, 185, 80] if a['label'] == 'Non_Damage' else [248, 81, 73]
                    map_data.append({
                        "lat": a['latitude'], "lon": a['longitude'],
                        "label": a['label'], "device": a.get('device_name', ''),
                        "color": color
                    })
            
            if map_data:
                df_map = pd.DataFrame(map_data)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    df_map,
                    get_position='[lon, lat]',
                    get_color='color',
                    get_radius=200,
                    pickable=True
                )
                view_state = pdk.ViewState(
                    latitude=df_map['lat'].mean(),
                    longitude=df_map['lon'].mean(),
                    zoom=10, pitch=0
                )
                r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{device}\n{label}"})
                st.pydeck_chart(r)
            else:
                st.info("No GPS data available yet. Send from mobile.")

# ── TAB 3: ALERT HISTORY ─────────────────────────────────────────────────────
with tab3:
    st.subheader("⚠️ Alert History")
    if HAS_DB:
        alerts = get_all_buffered()
        if alerts:
            df = pd.DataFrame(alerts)
            df = df[["timestamp", "device_name", "label", "confidence", "latitude", "longitude", "synced"]]
            df["synced"] = df["synced"].map({1: "✅", 0: "⏳"})
            df["confidence"] = df["confidence"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No alerts found.")

# ── TAB 4: CLOUD DB ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("☁️ Cloud Database")
    
    # Check AWS connection
    aws_connected = False
    if HAS_AWS:
        aws_connected = aws_manager.is_aws_configured()
    
    if aws_connected:
        st.markdown('<span class="badge badge-green">✅ AWS CONNECTION ESTABLISHED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-orange">❌ AWS CONNECTION NOT ESTABLISHED (No Credentials)</span>', unsafe_allow_html=True)

    if HAS_DB:
        stats = cloud_stats()
        sc1, sc2 = st.columns(2)
        sc1.metric("Total Cloud Records", stats["total_cloud_alerts"])
        sc2.metric("Pending Sync", stats["pending_sync"])
        
        if st.session_state.online_mode and aws_connected:
            recs = aws_manager.fetch_aws_alerts()
            if recs:
                st.markdown("### 🖼️ Images Stored in AWS S3")
                
                # Show thumbnails of images
                cols = st.columns(4)
                for idx, rec in enumerate(recs):
                    if rec.get("image_s3_url") and rec["image_s3_url"] != "None":
                        presigned = aws_manager.get_presigned_url(rec["image_s3_url"])
                        with cols[idx % 4]:
                            if presigned:
                                st.image(presigned, caption=f"{rec['label']} - {rec['device_name']}", use_container_width=True)
                            else:
                                st.error("Image link expired or inaccessible.")
                
                st.markdown("### 📋 DynamoDB Alert Logs")
                df_recs = pd.DataFrame(recs)[["timestamp", "device_name", "label", "confidence", "latitude", "longitude"]]
                df_recs["confidence"] = df_recs["confidence"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(df_recs, use_container_width=True)

# ── TAB 5: ARCHITECTURE ──────────────────────────────────────────────────────
with tab5:
    st.subheader("🌐 Three-Layer Architecture")
    st.code("""
┌─────────────────────────────────────────────────────────────┐
│ ☁️ CLOUD LAYER (AWS DynamoDB + S3)                          │
│ Permanent storage, synced only when online                  │
└───────────────────────▲─────────────────────────────────────┘
                        │ Auto-sync via Internet
┌───────────────────────▼─────────────────────────────────────┐
│ 🌫️ FOG LAYER (Dew Aggregator / IoT Server)                 │
│ Local inference (Ensemble CNN), Alert Buffer, Streamlit UI  │
└───────────────────────▲─────────────────────────────────────┘
                        │ POST /predict (Image + GPS)
┌───────────────────────▼─────────────────────────────────────┐
│ 📱 DEW LAYER (IoT Mobile Devices)                           │
│ Field devices capturing images + location in real-time      │
└─────────────────────────────────────────────────────────────┘
    """)
