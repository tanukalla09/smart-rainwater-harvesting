import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
start_date = datetime(2005, 1, 1)
end_date = datetime(2024, 12, 31)
catchment_area = 10  # m²
efficiency = 0.85

# Generate date range
dates = pd.date_range(start_date, end_date)

rainfall = []
for date in dates:
    # Seasonal rainfall simulation
    if date.month in [6, 7, 8, 9]:  # Monsoon
        mm = np.random.randint(10, 35)
    elif date.month in [3, 4, 5]:  # Summer
        mm = np.random.randint(0, 5)
    else:  # Winter/Post-monsoon
        mm = np.random.randint(0, 15)
    rainfall.append(mm)

# Calculate inflow liters
inflow = [round(mm * catchment_area * efficiency, 2) for mm in rainfall]

# Generate usage liters
usage = [np.random.randint(800, 1200) for _ in dates]

# Create DataFrame
df = pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "rainfall_mm": rainfall,
    "inflow_liters": inflow,
    "usage_liters": usage
})

# Save to CSV
df.to_csv("data.csv", index=False)
print(f"✅ Dataset generated: {len(df)} rows from {start_date.date()} to {end_date.date()}")

