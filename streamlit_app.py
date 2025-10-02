import streamlit as st
import json
import os
import subprocess
import pandas as pd
import datetime
import smtplib
from email.mime.text import MIMEText

st.set_page_config(page_title="Smart Rainwater Harvesting", layout="wide")

# ----------------------
# Sidebar Settings
# ----------------------
st.sidebar.header("🌍 Location & Forecast Settings")
city_name = st.sidebar.text_input("Enter City Name", "Chennai")
prediction_days = st.sidebar.slider("Prediction Range (Days)", 7, 30, 15)
email_alert = st.sidebar.text_input("Overflow Alert Email", "")

# ----------------------
# Helper: Send Email
# ----------------------
def send_email_alert(to_email, message):
    try:
        # Configure your SMTP server here
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "your_email@gmail.com"
        sender_password = "your_password"

        msg = MIMEText(message)
        msg["Subject"] = "⚠️ Rainwater System Overflow Alert"
        msg["From"] = sender_email
        msg["To"] = to_email

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        st.error(f"Email sending failed: {e}")

# ----------------------
# Load Forecast Button with Auto-Refresh
# ----------------------
if st.sidebar.button("🔍 Load Forecast"):
    results_file = f"{city_name.lower()}_results.json"

    def cache_is_fresh(file):
        if not os.path.exists(file):
            return False
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file))
        return (datetime.datetime.now() - file_time).days < 1

    if cache_is_fresh(results_file):
        st.sidebar.success(f"✅ Loaded cached forecast for {city_name}")
        with open(results_file) as f:
            results = json.load(f)
    else:
        st.sidebar.warning(f"⚠️ Cache outdated. Running ML pipeline...")
        subprocess.run(["python", "smart_rainwater_pipeline.py", city_name])
        with open(results_file) as f:
            results = json.load(f)

    # Slice data
    idx = prediction_days
    results["dates"] = results["dates"][:idx]
    results["predicted_rainfall"] = results["predicted_rainfall"][:idx]
    results["predicted_tank_levels"] = results["predicted_tank_levels"][:idx]
    results["daily_usage"] = results["daily_usage"][:idx]

    # ----------------------
    # Charts
    # ----------------------
    rain_df = pd.DataFrame({
        "Date": results["dates"],
        "Rainfall (Liters)": results["predicted_rainfall"]
    })
    st.subheader(f"🌧 Rainfall Prediction ({prediction_days} days)")
    st.bar_chart(rain_df, x="Date", y="Rainfall (Liters)")

    tank_df = pd.DataFrame({
        "Date": results["dates"],
        "Tank Level (Liters)": results["predicted_tank_levels"]
    })
    st.subheader(f"💧 Tank Levels ({prediction_days} days)")
    st.line_chart(tank_df, x="Date", y="Tank Level (Liters)")

    usage_df = pd.DataFrame({
        "Date": results["dates"],
        "Daily Usage (Liters)": results["daily_usage"]
    })
    st.subheader(f"📊 Optimized Daily Usage ({prediction_days} days)")
    st.bar_chart(usage_df, x="Date", y="Daily Usage (Liters)")

    # ----------------------
    # Monthly Impact Tracking
    # ----------------------
    st.subheader("📈 Monthly Impact Tracking")
    monthly_df = pd.DataFrame({
        "Month": pd.to_datetime(results["dates"]).to_period("M").astype(str),
        "Saved (Liters)": results["predicted_rainfall"]
    }).groupby("Month").sum().reset_index()

    st.line_chart(monthly_df, x="Month", y="Saved (Liters)")

    # ----------------------
    # Summary
    # ----------------------
    st.subheader("📊 Summary")
    st.write(f"**Total Rainwater Saved:** {results['total_rainwater_saved']} liters")
    st.write(f"**Total Savings:** Rs {results['total_rs_saved']:.2f}")
    st.write(f"**Efficiency:** {results['efficiency']}%")
    st.write(f"**Cost per Liter:** Rs {results['cost_per_liter']}")

    if not results["overflow_alerts"]:
        st.success("✅ No overflow alerts.")
    else:
        st.error(f"⚠️ Overflow Alerts: {results['overflow_alerts']}")
        if email_alert:
            send_email_alert(email_alert, f"Overflow Alerts: {results['overflow_alerts']}")

    # ----------------------
    # Help Section
    # ----------------------
    with st.expander("📘 How to Use"):
        st.write("""
        1. **Rainfall Chart** → Shows forecasted rainfall for upcoming days.
        2. **Tank Levels Chart** → Predicts water levels in your tank.
        3. **Daily Usage Chart** → Optimized usage schedule.
        4. **Monthly Impact** → Total liters saved per month.
        5. **Overflow Alerts** → If risk detected, reduce usage or increase storage.
        6. **Email Alerts** → Enter your email to get notified instantly.
        """)

