import yfinance as yf
import pandas as pd

tickers = {
    "Corn": "ZC=F",
    "Soybeans": "ZS=F", 
    "Wheat": "ZW=F",
    "Cattle": "LE=F",
    "Hogs": "HE=F",
    "Ethanol": "ETH=F",
    "NatGas": "NG=F", 
    "Cotton": "CT=F",
    "DBA": "DBA",
    "GSG": "GSG"
}

try:
    data = yf.download(list(tickers.values()), start="2020-01-01", end="2024-04-01")['Close']
    print(data.info())
    print("-" * 30)
    print("Null counts:")
    print(data.isnull().sum())
    print("-" * 30)
    print("Tail data:")
    print(data.tail())
except Exception as e:
    print(f"Error: {e}")
