# src/fdl_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# --------------------------- CONFIG --------------------------- #
st.set_page_config(page_title="🌍 FDL Disaster Dashboard", layout="wide")

LOG_FILE = "data/local_alerts.log"
METRICS_FILE = "data/training_metrics.csv"  # optional, created automatically
MODEL_INFO = {
    "Clients": 3,
    "Disaster Classes": 9,
    "Model": "CNN (PyTorch)",
    "Aggregator": "Dew Layer @ localhost:9090",
}

# --------------------------- HELPER FUNCTIONS --------------------------- #
def read_alerts():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["Timestamp", "Alert"])
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines[-20:]:
        ts, msg = line.strip().split(" - ", 1) if " - " in line else (datetime.now().strftime("%H:%M:%S"), line)
        data.append({"Timestamp": ts, "Alert": msg})
    return pd.DataFrame(data)

def read_metrics():
    if not os.path.exists(METRICS_FILE):
        return pd.DataFrame(columns=["Round", "Loss", "Accuracy"])
    return pd.read_csv(METRICS_FILE)

def plot_metrics(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if len(df) > 0:
        ax[0].plot(df["Round"], df["Loss"], marker="o")
        ax[0].set_title("Loss vs Round")
        ax[0].set_xlabel("Round")
        ax[0].set_ylabel("Loss")

        ax[1].plot(df["Round"], df["Accuracy"], marker="s", color="green")
        ax[1].set_title("Accuracy vs Round")
        ax[1].set_xlabel("Round")
        ax[1].set_ylabel("Accuracy")
    else:
        ax[0].text(0.5, 0.5, "No data yet", ha="center", va="center")
        ax[1].text(0.5, 0.5, "No data yet", ha="center", va="center")
    st.pyplot(fig)

# --------------------------- LAYOUT --------------------------- #
st.title("🌤️ Federated Dew Learning – Disaster Resilience Dashboard")
st.markdown("Monitor training, alerts, and client network in real-time.")

tab1, tab2, tab3 = st.tabs(["⚠️ Real-Time Alerts", "📊 Model Metrics", "🌐 Network Overview"])

# --------------------------- TAB 1: ALERTS --------------------------- #
with tab1:
    st.subheader("⚠️ Latest Disaster Alerts")
    alert_placeholder = st.empty()
    st.info("Streaming live alerts from local clients...")
    while True:
        df_alerts = read_alerts()
        alert_placeholder.dataframe(df_alerts[::-1], use_container_width=True)
        time.sleep(3)
        st.rerun()

# --------------------------- TAB 2: METRICS --------------------------- #
with tab2:
    st.subheader("📈 Federated Model Performance")
    metrics_df = read_metrics()
    if len(metrics_df) == 0:
        st.warning("No metrics found yet — training may still be running.")
    else:
        plot_metrics(metrics_df)

# --------------------------- TAB 3: NETWORK --------------------------- #
with tab3:
    st.subheader("🌐 System & Network Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Connected Clients", MODEL_INFO["Clients"])
        st.metric("Disaster Classes", MODEL_INFO["Disaster Classes"])
        st.metric("Model Architecture", MODEL_INFO["Model"])
    with col2:
        st.success("Dew Aggregator Active ✅")
        st.write(f"🛰️ {MODEL_INFO['Aggregator']}")
        st.info("Data shared only as model weights — privacy preserved.")
    st.markdown("---")
    st.caption("Federated Dew Learning (FDL) • Research Prototype © 2025")
