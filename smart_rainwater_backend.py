import json
import numpy as np

def convert_numpy(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    return obj

def run_forecast(
    city_name,
    dates,
    predicted_rainfall,
    predicted_tank_levels,
    daily_usage,
    total_rainwater_saved,
    total_rs_saved,
    efficiency,
    overflow_alerts,
    cost_per_liter,
    output_file="results.json"
):
    """
    Save forecast results to JSON file and print debug info.
    """

    # ✅ Print debug info
    print("\n================= 🛠 DEBUG: run_forecast() Received Parameters =================")
    print(f"🏙 City Name: {city_name}")
    print(f"📅 Dates: {dates}")
    print(f"🌧 Predicted Rainfall (Liters): {predicted_rainfall}")
    print(f"💧 Predicted Tank Levels (Liters): {predicted_tank_levels}")
    print(f"📊 Daily Usage (Liters): {daily_usage}")
    print(f"💦 Total Rainwater Saved (Liters): {total_rainwater_saved}")
    print(f"💰 Total Savings (Rs): {total_rs_saved}")
    print(f"⚙️ Efficiency (%): {efficiency}")
    print(f"🚨 Overflow Alerts: {overflow_alerts}")
    print(f"💰 Cost per Liter (Rs): {cost_per_liter}")
    print("================================================================\n")

    # ✅ Convert NumPy types to Python native types
    results = {
        "city_name": city_name,
        "dates": dates,
        "predicted_rainfall": convert_numpy(predicted_rainfall),
        "predicted_tank_levels": convert_numpy(predicted_tank_levels),
        "daily_usage": convert_numpy(daily_usage),
        "total_rainwater_saved": convert_numpy(total_rainwater_saved),
        "total_rs_saved": convert_numpy(total_rs_saved),
        "efficiency": convert_numpy(efficiency),
        "overflow_alerts": convert_numpy(overflow_alerts),
        "cost_per_liter": convert_numpy(cost_per_liter)
    }

    # ✅ Save to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Results saved to {output_file}")

def get_results(input_file="results.json"):
    """
    Load forecast results from JSON file.
    """
    with open(input_file, "r") as f:
        return json.load(f)
