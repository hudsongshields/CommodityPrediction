import requests
import pandas as pd
import time
from datetime import datetime
import os

def fetch_global_weather():
    """
    Automates the acquisition of historical daily meteorological data for global commodity hubs.
    
    The script interfaces with the Open-Meteo Historical Weather API to retrieve 
    surface-level atmospheric variables.
    
    API: https://archive-api.open-meteo.com/v1/archive
    """
    
    # Global Meteorological Hubs associated with primary commodity markets.
    locations = [
        # US Grains and Livestock (Midwest and Plains)
        {"city": "Des Moines", "country": "USA", "lat": 41.5868, "lon": -93.6250, "note": "Grains/Ethanol"},
        {"city": "Omaha", "country": "USA", "lat": 41.2565, "lon": -95.9345, "note": "Livestock"},
        {"city": "Cuiabá", "country": "Brazil", "lat": -15.6010, "lon": -56.0974, "note": "Soybean Production"},
        {"city": "Rosario", "country": "Argentina", "lat": -32.9442, "lon": -60.6505, "note": "Agri-Export Hub"},
        {"city": "Wichita", "country": "USA", "lat": 37.6872, "lon": -97.3301, "note": "Wheat Production"},
        {"city": "Rostov-on-Don", "country": "Russia", "lat": 47.2357, "lon": 39.7015, "note": "Russian Wheat Hub"},
        {"city": "Houston", "country": "USA", "lat": 29.7604, "lon": -95.3698, "note": "Energy Infrastructure"},
        {"city": "Pittsburgh", "country": "USA", "lat": 40.4406, "lon": -79.9959, "note": "Energy Production"},
        {"city": "Dubai", "country": "UAE", "lat": 25.2048, "lon": 55.2708, "note": "Energy Hub"},
        {"city": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298, "note": "Commodity Exchange Hub"},
        {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198, "note": "Global Shipping Hub"},
        {"city": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074, "note": "Regulatory/Demand Hub"},
        {"city": "Moscow", "country": "Russia", "lat": 55.7558, "lon": 37.6173, "note": "Energy/Wheat Hub"}
    ]

    start_date = "2019-01-01"
    end_date = "2024-12-31"
    
    # Meteorological variables extracted for spatiotemporal analysis.
    daily_vars = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "shortwave_radiation_sum", "relative_humidity_2m_max"
    ]
    
    all_city_data = []
    print(f"Initiating meteorological data ingestion for {len(locations)} locations.")
    print(f"Data window: {start_date} to {end_date}")
    
    for loc in locations:
        city, country, lat, lon = loc["city"], loc["country"], loc["lat"], loc["lon"]
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            f"&daily={','.join(daily_vars)}&temperature_unit=fahrenheit&timezone=auto"
        )
        
        try:
            print(f"Requesting: {city}, {country}...", end=" ", flush=True)
            
            # Implementation of exponential backoff for rate-limit management (HTTP 429).
            max_retries, retry_count, success = 3, 0, False
            while retry_count < max_retries and not success:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    daily = data.get("daily", {})
                    if daily:
                        df = pd.DataFrame(daily)
                        df["city"], df["country"] = city, country
                        all_city_data.append(df)
                        print("Status: Success")
                        success = True
                    else:
                        print("Status: No data returned")
                        break
                elif response.status_code == 429:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    print(f"Status: Rate Limited - Retrying in {wait_time}s...", end=" ", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"Status: Request Failed (Code {response.status_code})")
                    break
        except Exception as e:
            print(f"Status: Runtime Error ({str(e)})")
            
        time.sleep(0.5) 
        
    if all_city_data:
        full_df = pd.concat(all_city_data, ignore_index=True)
        # Persistent storage of the meteorological dataset.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_file = os.path.join(project_root, "data", "global_daily_weather.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        full_df.to_csv(output_file, index=False)
        print(f"Data ingestion complete. {len(full_df)} records saved to {output_file}.")
    else:
        print("Data collection failed. No records were stored.")

if __name__ == "__main__":
    fetch_global_weather()
