import pandas as pd
import re
from pandas.api.types import CategoricalDtype

def load_data(file_path):
    """Tải dữ liệu từ file CSV"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Làm sạch và chuyển đổi dữ liệu"""
    # Làm sạch cột Indicator
    df['Indicator'] = df['Indicator'].astype(str).str.strip()
    
    # Chuyển wide → long
    time_cols = df.columns[2:]
    df_long = df.melt(
        id_vars=['Indicator', 'StockID'],
        value_vars=time_cols,
        var_name='Period',
        value_name='Value'
    )
    
    # Làm sạch và chuyển Value sang số
    df_long['Value'] = (
        df_long['Value']
        .astype(str)
        .str.replace(',', '')
        .str.replace('\n', '')
        .replace('', pd.NA)
        .astype(float)
    )
    df_long.dropna(subset=['Value'], inplace=True)
    
    # Chuẩn hóa định dạng thời gian
    period_order = [
        'Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023',
        'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024'
    ]
    
    # Chuyển Period thành kiểu Categorical với thứ tự
    period_type = CategoricalDtype(categories=period_order, ordered=True)
    df_long['Period'] = df_long['Period'].astype(period_type)
    df_long = df_long.sort_values(['Period'])

    
    return df_long

import pandas as pd
import numpy as np

def load_stock_data(file_path):
    """Tải và tiền xử lý dữ liệu cổ phiếu"""
    df = pd.read_csv(file_path)
    
    # Tiền xử lý cơ bản
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    
    # Chuyển đổi các cột số
    numeric_cols = [
        "Total Volume", "Total Value", "Market Cap",
        "Closing Price", "Price Change", "Price Change %", 
        "Matched Volume", "Matched Value"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sắp xếp theo ngày
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df

def calculate_technical_indicators(df):
    """Tính toán các chỉ báo kỹ thuật"""
    df = df.copy()
    
    # SMA (Simple Moving Average)
    df['SMA_14'] = df['Closing Price'].rolling(window=14).mean()
    df['SMA_50'] = df['Closing Price'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    def compute_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI_14'] = compute_rsi(df['Closing Price'])
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Closing Price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Closing Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['BB_middle'] = df['Closing Price'].rolling(window=20).mean()
    df['BB_std'] = df['Closing Price'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    return df