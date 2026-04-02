import requests
import pandas as pd
import time
from datetime import datetime
import os

def fetch_global_weather():
    """
    Fetches historical daily weather data for key global commodity hubs and diverse cities.
    API: Open-Meteo Historical Weather (https://archive-api.open-meteo.com/v1/archive)
    """
    
    # 1. Define the Global Hubs (Focusing on the 8 core commodities: Corn, Soy, Wheat, Cattle, Hogs, Ethanol, Gas, Cotton)
    locations = [
        # --- US Core Grains & Ethanol (Midwest) ---
        {"city": "Des Moines", "country": "USA", "lat": 41.5868, "lon": -93.6250, "note": "Corn/Soy/Hogs/Ethanol"},
        {"city": "Omaha", "country": "USA", "lat": 41.2565, "lon": -95.9345, "note": "Grains/Livestock"},
        
        # --- Brazil & Argentina (Soy/Corn) ---
        {"city": "Cuiabá", "country": "Brazil", "lat": -15.6010, "lon": -56.0974, "note": "Mato Grosso Soy Hub"},
        {"city": "Rosario", "country": "Argentina", "lat": -32.9442, "lon": -60.6505, "note": "Grain Export Hub"},
        {"city": "São Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333, "note": "Sugar/Coffee/Finance"},
        
        # --- Global Wheat Belt ---
        {"city": "Wichita", "country": "USA", "lat": 37.6872, "lon": -97.3301, "note": "Wheat Hub"},
        {"city": "Rostov-on-Don", "country": "Russia", "lat": 47.2357, "lon": 39.7015, "note": "Russian Wheat Hub"},
        {"city": "Odessa", "country": "Ukraine", "lat": 46.4825, "lon": 30.7233, "note": "Black Sea Wheat Hub"},
        {"city": "Perth", "country": "Australia", "lat": -31.9505, "lon": 115.8605, "note": "Wheat Hub (WA)"},
        
        # --- Livestock Hubs ---
        {"city": "Amarillo", "country": "USA", "lat": 35.2220, "lon": -101.8313, "note": "Cattle Hub"},
        {"city": "Brisbane", "country": "Australia", "lat": -27.4698, "lon": 153.0251, "note": "Cattle Hub (QLD)"},
        
        # --- Energy Centers (NatGas) ---
        {"city": "Houston", "country": "USA", "lat": 29.7604, "lon": -95.3698, "note": "Energy Hub"},
        {"city": "Pittsburgh", "country": "USA", "lat": 40.4406, "lon": -79.9959, "note": "Marcellus Shale Hub"},
        {"city": "Dubai", "country": "UAE", "lat": 25.2048, "lon": 55.2708, "note": "Energy Hub"},
        
        # --- Global Cotton & Softs ---
        {"city": "Lubbock", "country": "USA", "lat": 33.5779, "lon": -101.8552, "note": "Cotton Hub"},
        {"city": "Ahmedabad", "country": "India", "lat": 23.0225, "lon": 72.5714, "note": "Cotton Hub (Gujarat)"},
        {"city": "Urumqi", "country": "China", "lat": 43.8256, "lon": 87.6168, "note": "Cotton Hub (Xinjiang)"},
        {"city": "Nairobi", "country": "Kenya", "lat": -1.2921, "lon": 36.8219, "note": "Softs (Coffee/Tea)"},
        
        # --- Global Major Consumption & Financial Hubs ---
        {"city": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060, "note": "Finance"},
        {"city": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298, "note": "CME/Grain Trading"},
        {"city": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278, "note": "Finance/Commodities"},
        {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503, "note": "Global Demand"},
        {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198, "note": "Oil/Grains Hub"},
        {"city": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777, "note": "Global Demand"},
        {"city": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074, "note": "Global Demand"},
        {"city": "Lagos", "country": "Nigeria", "lat": 6.5244, "lon": 3.3792, "note": "Regional Demand"},
        {"city": "Moscow", "country": "Russia", "lat": 55.7558, "lon": 37.6173, "note": "Energy/Wheat Hub"},
        {"city": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357, "note": "Major Importer Hub"},
        {"city": "Buenos Aires", "country": "Argentina", "lat": -34.6037, "lon": -58.3816, "note": "Agriculture/Fin"},
        {"city": "Johannesburg", "country": "South Africa", "lat": -26.2041, "lon": 28.0473, "note": "Corn Hub (RSA)"}
    ]

    start_date = "2019-01-01"
    end_date = "2024-12-31"
    
    daily_vars = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min",
        "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
        "weather_code", "windspeed_10m_max", "windgusts_10m_max",
        "winddirection_10m_dominant", "shortwave_radiation_sum",
        "relative_humidity_2m_max", "relative_humidity_2m_min"
    ]
    
    all_city_data = []
    
    print(f"🚀 Starting Weather Data Ingestion for {len(locations)} global locations...")
    print(f"📅 Period: {start_date} to {end_date}")
    print("-" * 50)
    
    for loc in locations:
        city = loc["city"]
        country = loc["country"]
        lat = loc["lat"]
        lon = loc["lon"]
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            f"&daily={','.join(daily_vars)}&temperature_unit=fahrenheit&timezone=auto"
        )
        
        try:
            print(f"📥 Fetching: {city}, {country} ({loc['note']})...", end=" ", flush=True)
            
            # Implementation of Retries for 429s (Polite Backoff)
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    daily = data.get("daily", {})
                    
                    if daily:
                        df = pd.DataFrame(daily)
                        df["city"] = city
                        df["country"] = country
                        df["latitude"] = lat
                        df["longitude"] = lon
                        
                        all_city_data.append(df)
                        print("✅ Success")
                        success = True
                    else:
                        print("⚠️ No daily data found.")
                        break
                elif response.status_code == 429:
                    retry_count += 1
                    wait_time = 2 ** retry_count # Exponential backoff
                    print(f"🔄 429 (Limit) - Waiting {wait_time}s...", end=" ", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed (Status: {response.status_code})")
                    break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        time.sleep(1.0) # Increased polite delay
        
    if all_city_data:
        full_df = pd.concat(all_city_data, ignore_index=True)
        
        # Save to CSV
        output_file = "global_daily_weather.csv"
        full_df.to_csv(output_file, index=False)
        
        print("-" * 50)
        print("📁 WEATHER DATASET SUMMARY")
        print("-" * 50)
        print(f"Total Rows Saved: {len(full_df)}")
        print(f"Locations Gathered: {full_df['city'].nunique()}")
        print(f"Date Range: {full_df['time'].min()} to {full_df['time'].max()}")
        print(f"Variables Extracted: {len(daily_vars)}")
        print(f"Output File: {os.path.abspath(output_file)}")
        print("-" * 50)
    else:
        print("🚨 No data was collected. Please check the API status or your internet connection.")

if __name__ == "__main__":
    fetch_global_weather()
