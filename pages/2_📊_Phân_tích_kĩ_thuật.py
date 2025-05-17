import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_stock_data, calculate_technical_indicators

# ======= 1. Thiết lập trang =======
st.set_page_config(layout="wide")
st.title("📈 Phân tích Kỹ thuật Chuyên sâu")

# ======= 2. Tải và xử lý dữ liệu =======
@st.cache_data
def load_all_data():
    df_fpt = load_stock_data("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
    df_cmg = load_stock_data("4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv")
    
    df_fpt = calculate_technical_indicators(df_fpt)
    df_cmg = calculate_technical_indicators(df_cmg)
    
    return df_fpt, df_cmg

df_fpt, df_cmg = load_all_data()

# ======= 3. Chọn cổ phiếu và khoảng thời gian =======
col1, col2, col3 = st.columns([1,1,2])
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

# ======= 4. Tạo các tab phân tích =======
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "TỔNG QUAN", 
    "ĐƯỜNG TRUNG BÌNH", 
    "BOLLINGER BANDS", 
    "MACD", 
    "RSI"
])

with tab1:
    # ======= Tab tổng quan =======
    st.subheader(f"Tổng quan {selected_stock} ({days_to_show} ngày)")
    
    # Biểu đồ giá đóng cửa và volume
    fig_overview = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Biểu đồ giá
    fig_overview.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Closing Price'],
            name='Giá đóng cửa',
            line=dict(color='#0E6994', width=2)
        ),
        row=1, col=1
    )
    
    # Biểu đồ volume
    fig_overview.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Total Volume'],
            name='Khối lượng',
            marker_color='#7EC8E3',
            opacity=0.6
        ),
        row=2, col=1
    )
    
    fig_overview.update_layout(
        height=600,
        showlegend=True,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_overview, use_container_width=True)
    
    # Thông tin tổng quan
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá hiện tại", f"{df['Closing Price'].iloc[-1]:,.2f}")
    with col2:
        change = df['Closing Price'].iloc[-1] - df['Closing Price'].iloc[-2]
        pct_change = (change / df['Closing Price'].iloc[-2]) * 100
        st.metric("Thay đổi", f"{change:,.2f}", f"{pct_change:.2f}%")
    with col3:
        st.metric("Khối lượng TB", f"{df['Total Volume'].mean():,.0f}")

with tab2:
    # ======= Tab đường trung bình =======
    st.subheader("Phân tích Đường Trung bình")
    
    fig_ma = go.Figure()
    
    # Giá đóng cửa
    fig_ma.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Closing Price'],
            name='Giá đóng cửa',
            line=dict(color='#0E6994', width=1.5)
        )
    )
    
    # Các đường trung bình
    fig_ma.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['SMA_14'],
            name='SMA 14 ngày',
            line=dict(color='#FD6200', width=2)
        )
    )
    
    fig_ma.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['EMA_12'],
            name='EMA 12 ngày',
            line=dict(color='#D8A100', width=2)
        )
    )
    
    fig_ma.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['EMA_26'],
            name='EMA 26 ngày',
            line=dict(color='#A05195', width=2)
        )
    )
    
    fig_ma.update_layout(
        height=500,
        hovermode="x unified",
        yaxis_title="Giá",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # Giải thích tín hiệu
    with st.expander("📌 Giải thích tín hiệu"):
        st.markdown("""
        - **SMA (Simple Moving Average)**: Đường trung bình động đơn giản
        - **EMA (Exponential Moving Average)**: Đường trung bình động hàm mũ, nhạy cảm hơn với biến động gần
        - **Tín hiệu mua**: Khi đường ngắn (EMA12) cắt lên đường dài (EMA26)
        - **Tín hiệu bán**: Khi đường ngắn cắt xuống đường dài
        """)

with tab3:
    # ======= Tab Bollinger Bands =======
    st.subheader("Bollinger Bands")
    
    fig_bb = go.Figure()
    
    # Bollinger Bands
    fig_bb.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_upper'],
            name='Dải trên',
            line=dict(color='#84D0D0', width=1),
            fill=None
        )
    )
    
    fig_bb.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_lower'],
            name='Dải dưới',
            line=dict(color='#84D0D0', width=1),
            fill='tonexty'
        )
    )
    
    fig_bb.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_middle'],
            name='Dải giữa',
            line=dict(color='#FD6200', width=2)
        )
    )
    
    # Giá đóng cửa
    fig_bb.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Closing Price'],
            name='Giá đóng cửa',
            line=dict(color='#0E6994', width=2)
        )
    )
    
    fig_bb.update_layout(
        height=500,
        hovermode="x unified",
        yaxis_title="Giá",
        showlegend=True
    )
    
    st.plotly_chart(fig_bb, use_container_width=True)
    
    # Giải thích tín hiệu
    with st.expander("📌 Giải thích tín hiệu"):
        st.markdown("""
        - **Dải Bollinger**: Đo lường biến động giá
        - **Giá chạm dải trên**: Có thể quá mua (overbought)
        - **Giá chạm dải dưới**: Có thể quá bán (oversold)
        - **Bóp dải**: Chuẩn bị có biến động mạnh
        """)

with tab4:
    # ======= Tab MACD =======
    st.subheader("MACD (Moving Average Convergence Divergence)")
    
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.02, row_heights=[0.7, 0.3])
    
    # Biểu đồ giá
    fig_macd.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Closing Price'],
            name='Giá đóng cửa',
            line=dict(color='#0E6994', width=2)
        ),
        row=1, col=1
    )
    
    # Biểu đồ MACD
    fig_macd.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MACD'],
            name='MACD',
            line=dict(color='#0E6994', width=2)
        ),
        row=2, col=1
    )
    
    fig_macd.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Signal_Line'],
            name='Đường tín hiệu',
            line=dict(color='#FD6200', width=2)
        ),
        row=2, col=1
    )
    
    fig_macd.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Histogram'],
            name='Histogram',
            marker_color=['green' if x >=0 else 'red' for x in df['Histogram']]
        ),
        row=2, col=1
    )
    
    fig_macd.update_layout(
        height=700,
        hovermode="x unified",
        showlegend=True
    )
    
    st.plotly_chart(fig_macd, use_container_width=True)
    
    # Giải thích tín hiệu
    with st.expander("📌 Giải thích tín hiệu"):
        st.markdown("""
        - **MACD > Signal**: Tín hiệu tăng giá
        - **MACD < Signal**: Tín hiệu giảm giá
        - **Histogram tăng**: Động lực tăng đang mạnh lên
        - **Histogram giảm**: Động lực tăng đang yếu đi
        """)

with tab5:
    # ======= Tab RSI =======
    st.subheader("RSI (Relative Strength Index)")
    
    fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.02, row_heights=[0.7, 0.3])
    
    # Biểu đồ giá
    fig_rsi.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Closing Price'],
            name='Giá đóng cửa',
            line=dict(color='#0E6994', width=2)
        ),
        row=1, col=1
    )
    
    # Biểu đồ RSI
    fig_rsi.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['RSI_14'],
            name='RSI 14 ngày',
            line=dict(color='#A05195', width=2)
        ),
        row=2, col=1
    )
    
    # Thêm các đường mức
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig_rsi.update_layout(
        height=700,
        hovermode="x unified",
        showlegend=True,
        yaxis2=dict(range=[0,100], title="RSI")
    )
    
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Giải thích tín hiệu
    with st.expander("📌 Giải thích tín hiệu"):
        st.markdown("""
        - **RSI > 70**: Quá mua, có thể điều chỉnh giảm
        - **RSI < 30**: Quá bán, có thể phục hồi
        - **Phân kỳ RSI**: Cảnh báo đảo chiều tiềm năng
        """)

# ======= 5. Bảng dữ liệu =======
with st.expander("📊 Xem dữ liệu chi tiết"):
    st.dataframe(
        df.sort_values('Date', ascending=False),
        use_container_width=True,
        height=400
    )