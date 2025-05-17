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