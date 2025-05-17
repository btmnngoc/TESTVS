import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_stock_data

# ======= 1. Thiết lập trang =======
st.set_page_config(layout="wide")
st.title("📊 Phân tích Kỹ thuật Giá cổ phiếu")

# ======= 2. Tải dữ liệu =======
@st.cache_data
def load_data():
    df_fpt = load_stock_data("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
    df_cmg = load_stock_data("4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv")
    return df_fpt, df_cmg

df_fpt, df_cmg = load_data()

# ======= 3. Chọn cổ phiếu và khoảng thời gian =======
col1, col2 = st.columns(2)
with col1:
    selected_stock = st.selectbox(
        "Chọn cổ phiếu",
        ["FPT", "CMG"],
        index=0
    )
    
with col2:
    days_to_show = st.slider(
        "Số ngày hiển thị",
        min_value=30,
        max_value=365,
        value=90,
        step=30
    )

df = df_fpt if selected_stock == "FPT" else df_cmg
df = df.tail(days_to_show).copy()

# ======= 4. Tính toán chỉ báo kỹ thuật =======
def calculate_technical_indicators(df):
    # SMA
    df['SMA_14'] = df['Closing Price'].rolling(window=14).mean()
    df['SMA_50'] = df['Closing Price'].rolling(window=50).mean()
    
    # RSI
    def compute_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI_14'] = compute_rsi(df['Closing Price'])
    
    # MACD
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

df = calculate_technical_indicators(df)

# ======= 5. Vẽ biểu đồ =======
tab1, tab2, tab3 = st.tabs(["Biểu đồ nến", "Chỉ báo kỹ thuật", "Tổng hợp"])

with tab1:
    # Biểu đồ nến
    fig_candlestick = go.Figure(
        data=[
            go.Candlestick(
                x=df['Date'],
                open=df['Opening Price'],
                high=df['Highest Price'],
                low=df['Lowest Price'],
                close=df['Closing Price'],
                name='Giá'
            )
        ]
    )
    
    fig_candlestick.update_layout(
        title=f'Biểu đồ nến {selected_stock}',
        xaxis_title='Ngày',
        yaxis_title='Giá',
        height=600
    )
    st.plotly_chart(fig_candlestick, use_container_width=True)

with tab2:
    # Chọn chỉ báo kỹ thuật
    indicator = st.selectbox(
        "Chọn chỉ báo",
        ["MACD", "RSI", "Bollinger Bands", "Đường trung bình"],
        index=0
    )
    
    if indicator == "MACD":
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], name='Signal Line', line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=df['Date'], y=df['Histogram'], name='Histogram', 
                                marker_color=['green' if x >=0 else 'red' for x in df['Histogram']]))
        fig_macd.update_layout(title='MACD', height=400)
        st.plotly_chart(fig_macd, use_container_width=True)
        
    elif indicator == "RSI":
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], name='RSI 14', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title='RSI (14 ngày)', yaxis_range=[0,100], height=400)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    elif indicator == "Bollinger Bands":
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_upper'], name='Upper Band', line=dict(color='gray')))
        fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_lower'], name='Lower Band', line=dict(color='gray'), fill='tonexty'))
        fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_middle'], name='Middle Band', line=dict(color='blue')))
        fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Closing Price'], name='Giá đóng cửa', line=dict(color='red')))
        fig_bb.update_layout(title='Bollinger Bands', height=400)
        st.plotly_chart(fig_bb, use_container_width=True)
        
    else:  # Đường trung bình
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Closing Price'], name='Giá đóng cửa', line=dict(color='black')))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_14'], name='SMA 14', line=dict(color='blue')))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='red')))
        fig_ma.update_layout(title='Đường trung bình', height=400)
        st.plotly_chart(fig_ma, use_container_width=True)

with tab3:
    # Biểu đồ tổng hợp
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3])
    
    # Biểu đồ nến
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Opening Price'],
        high=df['Highest Price'],
        low=df['Lowest Price'],
        close=df['Closing Price'],
        name='Giá'
    ), row=1, col=1)
    
    # Đường trung bình
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['SMA_14'],
        name='SMA 14',
        line=dict(color='blue', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='red', width=1)
    ), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Histogram'],
        name='Histogram',
        marker_color=['green' if x >=0 else 'red' for x in df['Histogram']]
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Signal_Line'],
        name='Signal Line',
        line=dict(color='orange', width=1)
    ), row=2, col=1)
    
    fig.update_layout(
        title=f'Phân tích kỹ thuật tổng hợp - {selected_stock}',
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)