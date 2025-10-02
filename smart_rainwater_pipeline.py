import sys
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import random
from smart_rainwater_backend import run_forecast

# -------------------------------
# Get Coordinates
# -------------------------------
def get_coordinates(city_name, api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={city_name}&key={api_key}"
    r = requests.get(url)
    data = r.json()
    if not data['results']:
        raise ValueError("❌ No coordinates found.")
    return data['results'][0]['geometry']['lat'], data['results'][0]['geometry']['lng']

# -------------------------------
# Water Tariff (Cached)
# -------------------------------
def get_water_tariff(city_name):
    cache_file = "tariff_cache.json"
    default_tariff = 0.05
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cache_data = json.load(f)
            if city_name.lower() in cache_data:
                return cache_data[city_name.lower()]["cost_per_liter"]
        except:
            pass
    print(f"⚠️ Using default Rs {default_tariff}/liter")
    return default_tariff

# -------------------------------
# Rainfall Data (Last 2 Years Only)
# -------------------------------
def get_rainfall(lat, lon, city_name):
    master_file = "rainfall_cache_all_cities.csv"
    end_year = datetime.now().year
    start_year = end_year - 2

    if os.path.exists(master_file):
        df_all = pd.read_csv(master_file)
        if city_name in df_all["city"].unique():
            print(f"🌦 Loaded rainfall data for {city_name} from cache")
            return df_all[df_all["city"] == city_name]

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOT&community=RE&longitude={lon}&latitude={lat}"
        f"&start={start_year}0101&end={end_year}1231"
        f"&format=CSV&header=true&user=anonymous"
    )
    r = requests.get(url)
    if r.status_code != 200:
        raise ConnectionError(f"❌ NASA API error: {r.text}")

    with open("rainfall_temp.csv", "wb") as f:
        f.write(r.content)

    with open("rainfall_temp.csv") as f:
        lines = f.readlines()
    data_start = next(i for i, line in enumerate(lines) if line.strip().startswith("YEAR"))
    df = pd.read_csv("rainfall_temp.csv", skiprows=data_start)
    if 'PRECTOTCORR' in df.columns:
        df.rename(columns={'PRECTOTCORR': 'PRECTOT'}, inplace=True)
    df["city"] = city_name
    df.rename(columns={"YEAR": "year", "MO": "month", "DY": "day"}, inplace=True)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]]).dt.strftime("%Y-%m-%d")

    if os.path.exists(master_file):
        df_all = pd.read_csv(master_file)
        df_all = df_all[df_all["city"] != city_name]
        df_all = pd.concat([df_all, df], ignore_index=True)
    else:
        df_all = df
    df_all.to_csv(master_file, index=False)
    return df

# -------------------------------
# Data Cleaning & Tank Calculations
# -------------------------------
def clean_missing_data(df):
    return df[df['PRECTOT'] != -999]

def calculate_inflow(df, catchment_area=120, efficiency=0.85):
    df['Inflow_Liters'] = df['PRECTOT'] * catchment_area * efficiency
    return df

def add_usage(df, daily_usage=500):
    df['Usage_Liters'] = daily_usage
    return df

def calculate_storage(df, tank_capacity=5000):
    df['Net_Storage'] = df['Inflow_Liters'] - df['Usage_Liters']
    tank_level = []
    current_level = 0
    for change in df['Net_Storage']:
        current_level += change
        current_level = max(0, min(current_level, tank_capacity))
        tank_level.append(current_level)
    df['Tank_Level'] = tank_level
    return df

# -------------------------------
# Train or Load Cached LSTM (.keras format)
# -------------------------------
def train_or_load_lstm(city, df, column_name, sequence_length=30):
    model_file = f"{city.lower()}_{column_name}_lstm_model.keras"
    scaler_file = f"{city.lower()}_{column_name}_scaler.npy"

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[column_name]])

    np.save(scaler_file, scaler.data_min_)
    np.save(scaler_file.replace(".npy", "_max.npy"), scaler.data_max_)

    if os.path.exists(model_file):
        print(f"⚡ Loaded cached LSTM model for {city} ({column_name})")
        model = load_model(model_file, compile=False)  # <-- FIX: skip compile
    else:
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length])
        X, y = np.array(X), np.array(y)

        model = Sequential()
        model.add(Input(shape=(sequence_length, 1)))  # <-- FIX: proper Input layer
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        model.save(model_file)  # <-- FIX: save in native format

    last_sequence = scaled_data[-sequence_length:]
    predictions = []
    current_seq = last_sequence.reshape(1, sequence_length, 1)
    for _ in range(30):
        pred = model.predict(current_seq, verbose=0)[0][0]
        predictions.append(pred)
        current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

    predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    predictions_rescaled = [max(0, val) for val in predictions_rescaled]
    return predictions_rescaled

# -------------------------------
# GA Optimization
# -------------------------------
def genetic_algorithm_optimize(predicted_levels, inflow_forecast, tank_capacity=5000, min_usage=300, max_usage=800):
    days = len(predicted_levels)
    def fitness(schedule):
        tank = predicted_levels[0]
        penalty, savings = 0, 0
        for day in range(days):
            tank += inflow_forecast[day] - schedule[day]
            if tank > tank_capacity:
                penalty += (tank - tank_capacity) * 2
                tank = tank_capacity
            if tank < 0:
                penalty += abs(tank) * 3
                tank = 0
            savings += schedule[day]
        return savings - penalty

    population = [[random.randint(min_usage, max_usage) for _ in range(days)] for _ in range(20)]
    for _ in range(50):
        scores = [fitness(ind) for ind in population]
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        population = [ind for _, ind in ranked[:10]]
        while len(population) < 20:
            p1, p2 = random.sample(population[:5], 2)
            cut = random.randint(1, days-1)
            child = p1[:cut] + p2[cut:]
            if random.random() < 0.2:
                child[random.randint(0, days-1)] = random.randint(min_usage, max_usage)
            population.append(child)
    return max(population, key=fitness)

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    api_key = "443850e596e649d08170c4020eda4961"
    city = sys.argv[1] if len(sys.argv) > 1 else "Chennai"

    results_file = f"{city.lower()}_results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        run_forecast(**results)
        sys.exit()

    lat, lon = get_coordinates(city, api_key)
    cost_per_liter = get_water_tariff(city)
    rain_df = get_rainfall(lat, lon, city)
    rain_df = clean_missing_data(rain_df)
    rain_df = calculate_inflow(rain_df)
    rain_df = add_usage(rain_df)
    rain_df = calculate_storage(rain_df)

    predictions_30d = train_or_load_lstm(city, rain_df, "Tank_Level")
    rainfall_predictions_30d = train_or_load_lstm(city, rain_df, "Inflow_Liters")

    inflow_forecast = list(rain_df['Inflow_Liters'].tail(30).values)
    inflow_forecast = [max(0, val) for val in inflow_forecast]

    best_usage_schedule = genetic_algorithm_optimize(predictions_30d, inflow_forecast)

    forecast_dates = [(datetime.now().date() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(30)]

    results = dict(
        city_name=city,
        dates=forecast_dates,
        predicted_rainfall=list(map(float, rainfall_predictions_30d)),
        predicted_tank_levels=list(map(float, predictions_30d)),
        daily_usage=list(map(float, best_usage_schedule)),
        total_rainwater_saved=float(sum(inflow_forecast)),
        total_rs_saved=float(sum(inflow_forecast) * cost_per_liter),
        efficiency=85.0,
        overflow_alerts=[],
        cost_per_liter=float(cost_per_liter)
    )

    with open(results_file, "w") as f:
        json.dump(results, f)

    run_forecast(**results)
