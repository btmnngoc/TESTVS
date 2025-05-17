import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_stock_data

# ======= 1. Thiết lập trang =======
st.set_page_config(layout="wide")
st.title("📊 Phân tích Biến động Giá cổ phiếu")

# ======= 2. Tải dữ liệu =======
@st.cache_data
def load_data():
    df_fpt = load_stock_data("4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv")
    df_cmg = load_stock_data("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
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

# ======= 4. Tính toán chỉ số biến động =======
def calculate_price_changes(df):
    # Tính biến động giá từng ngày
    df['Daily_Change'] = df['Price Change']
    df['Daily_Change_Pct'] = df['Price Change %']
    
    # Tính tổng tích lũy
    df['Cumulative_Change'] = df['Daily_Change'].cumsum()
    df['Cumulative_Change_Pct'] = (1 + df['Daily_Change_Pct']/100).cumprod() - 1
    
    return df

df = calculate_price_changes(df)

# ======= 5. Vẽ biểu đồ thác nước =======
st.subheader(f"Biến động giá {selected_stock} ({days_to_show} ngày gần nhất)")

# Tạo biểu đồ thác nước cho biến động giá
fig_waterfall = go.Figure(go.Waterfall(
    name="Biến động giá",
    x=df['Date'],
    y=df['Daily_Change'],
    textposition="outside",
    text=df['Daily_Change'].round(2),
    connector={"line":{"color":"rgb(63, 63, 63)"}},
))

fig_waterfall.update_layout(
    title=f"Biến động giá hàng ngày - {selected_stock}",
    xaxis_title="Ngày",
    yaxis_title="Thay đổi giá",
    showlegend=True,
    height=600
)

# Thêm đường tổng tích lũy
fig_waterfall.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Cumulative_Change'],
    name="Tổng tích lũy",
    line=dict(color='red', width=2),
    mode='lines+markers'
))

st.plotly_chart(fig_waterfall, use_container_width=True)

# ======= 6. Thông tin bổ sung =======
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Giá đóng cửa hiện tại",
        value=f"{df['Closing Price'].iloc[-1]:,.2f}",
        delta=f"{df['Daily_Change'].iloc[-1]:,.2f} ({df['Daily_Change_Pct'].iloc[-1]:.2f}%)"
    )

with col2:
    st.metric(
        label="Tổng biến động",
        value=f"{df['Cumulative_Change'].iloc[-1]:,.2f}",
        delta=f"{df['Cumulative_Change_Pct'].iloc[-1]*100:.2f}%"
    )

# ======= 7. Bảng dữ liệu chi tiết =======
with st.expander("📊 Xem dữ liệu chi tiết"):
    st.dataframe(
        df[['Date', 'Closing Price', 'Daily_Change', 'Daily_Change_Pct', 'Cumulative_Change', 'Cumulative_Change_Pct']]
        .style.format({
            'Closing Price': '{:,.2f}',
            'Daily_Change': '{:,.2f}',
            'Daily_Change_Pct': '{:.2f}%',
            'Cumulative_Change': '{:,.2f}',
            'Cumulative_Change_Pct': '{:.2f}%'
        })
        .background_gradient(subset=['Daily_Change', 'Daily_Change_Pct'], cmap='RdYlGn'),
        use_container_width=True,
        height=400
    )