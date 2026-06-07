# src/dashboard_app.py
import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="Dew-FDL Disaster Dashboard", layout="wide")

st.title("🌍 Disaster-Resilient Dew Federated Learning Dashboard")
st.markdown("---")

LOG_FILE = "data/local_alerts.log"

col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Real-time Disaster Alerts")
    if not os.path.exists(LOG_FILE):
        st.warning("No alerts yet... Waiting for client activity.")
    else:
        placeholder = st.empty()
        while True:
            try:
                with open(LOG_FILE, "r") as f:
                    logs = f.readlines()[-10:]  # show latest 10 alerts
                df = pd.DataFrame(logs, columns=["Recent Alerts"])
                placeholder.dataframe(df, use_container_width=True)
                time.sleep(2)
            except Exception:
                time.sleep(2)
                continue

with col2:
    st.subheader("📊 System Status")
    st.write("✅ Clients Connected: 3")
    st.write("☁️ Dew Aggregator: Active on port 9090")
    st.write("📦 Dataset: 9 Disaster Classes")
    st.write("⚙️ Global Model: CNN (PyTorch)")

st.markdown("---")
st.caption("Built with Streamlit • Federated Dew Learning © 2025")
