import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_stock_data

# ======= 1. Thiết lập trang =======
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        @keyframes gradient {
            0% { background-position: 0% center; }
            50% { background-position: 100% center; }
            100% { background-position: 0% center; }
        }
        
        .gradient-text {
            background: linear-gradient(90deg, #8A2BE2 0%, #00BFFF 50%, #7CFC00 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-size: 200% auto;
            animation: gradient 3s ease infinite;
        }
        
        .gradient-bar {
            background: linear-gradient(90deg, #8A2BE2, #00BFFF, #7CFC00);
            background-size: 200% auto;
            animation: gradient 3s ease infinite;
            height: 4px;
            width: 192px;
            margin: 0 auto;
            border-radius: 9999px;
        }
        
        .feature-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(138, 43, 226, 0.3);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 2rem;
            transition: all 0.3s ease;
            margin: 0.5rem; /* Thêm margin để cách nhau */
        }
        
        .feature-card:hover {
            transform: translateY(-5px) scale(1.02);
            border-color: #00BFFF;
            box-shadow: 0 10px 30px rgba(0, 191, 255, 0.3);
        }
        
        .welcome-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(0, 191, 255, 0.3);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 2rem 3rem;
            max-width: 64rem;
            margin: 0 auto 3rem auto;
        }
        
        body {
            background-color: #020617;
            color: #F8FAFC;
            font-family: 'Inter', sans-serif;
        }
        
        .header-bg {
            background: linear-gradient(135deg, #020617 0%, #0F172A 100%);
            box-shadow: 0 4px 30px rgba(0, 191, 255, 0.1);
            padding: 4rem;
            border-radius: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            text-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
        }
        
        .floating {
            animation: float 3s ease-in-out infinite;
        }
        
        .glow {
            filter: drop-shadow(0 0 8px currentColor);
        }
        
        .gradient-button {
            background: linear-gradient(90deg, #8A2BE2 0%, #00BFFF 50%, #7CFC00 100%);
            background-size: 200% auto;
            transition: all 0.3s ease;
            border-radius: 9999px;
            padding: 0.75rem 2rem;
            color: white;
            font-weight: 600;
            display: inline-block;
        }
        
        .gradient-button:hover {
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 10px 20px rgba(0, 191, 255, 0.4);
            background-position: right center;
        }
        
        .footer-bg {
            background: linear-gradient(180deg, rgba(2, 6, 23, 0) 0%, #020617 100%);
            padding: 3rem 0;
            margin-top: 6rem;
        }
        
        .card-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 2rem; /* Tăng khoảng cách giữa các thẻ */
            padding: 1rem;
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@700&display=swap');
    </style>
""", unsafe_allow_html=True)

# Tiêu đề trang
st.markdown("""
    <header class="header-bg">
        <div class="text-center">
            <div class="flex justify-center mb-6">
                <div class="relative">
                    <i class="fas fa-chart-line text-6xl gradient-text glow floating"></i>
                    <i class="fas fa-star text-xl text-yellow-400 absolute -top-2 -right-2"></i>
                </div>
            </div>
            <h1 class="main-title text-5xl font-bold mb-4 gradient-text">DABAFIN - PHÂN TÍCH KĨ THUẬT CỔ PHIẾU</h1>
            <p class="text-xl text-slate-300 mb-6">Phân tích, đánh giá các chỉ số kĩ thuật giá cổ phiếu</p>
            <div class="gradient-bar"></div>
        </div>
    </header>
""", unsafe_allow_html=True)

# ======= 2. Tải và xử lý dữ liệu =======
@st.cache_data
def load_and_process_data():
    df_fpt = load_stock_data("4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv")
    df_cmg = load_stock_data("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
    
    # Tính toán chỉ báo kỹ thuật cho cả 2 cổ phiếu
    for df in [df_fpt, df_cmg]:
        # Tính biến động giá
        df['Daily_Change'] = df['Price Change']
        df['Daily_Change_Pct'] = df['Price Change %']
       # Thay đổi cách tính cumulative change để đồng bộ với giá đóng cửa
        df['Cumulative_Change'] = df['Daily_Change'].cumsum()        
        df['Cumulative_Change_Pct'] = (1 + df['Daily_Change_Pct']/100).cumprod() - 1
        
        # Chỉ báo kỹ thuật
        df['SMA_14'] = df['Closing Price'].rolling(window=14).mean()
        df['EMA_12'] = df['Closing Price'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Closing Price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Bollinger Bands
        sma_20 = df['Closing Price'].rolling(window=20).mean()
        std_20 = df['Closing Price'].rolling(window=20).std()
        df['BB_upper'] = sma_20 + 2 * std_20
        df['BB_lower'] = sma_20 - 2 * std_20
        df['BB_middle'] = sma_20
        
        # RSI
        delta = df['Closing Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
    
    return df_fpt, df_cmg

df_fpt, df_cmg = load_and_process_data()

# ======= 3. Giao diện người dùng =======
col1, col2, col3 = st.columns([1,1,2])
with col1:
    selected_stock = st.selectbox("Chọn cổ phiếu", ["FPT", "CMG"], index=0)
    
with col2:
    days_to_show = st.slider("Số ngày hiển thị", 30, 365, 90, 30)

df = df_fpt if selected_stock == "FPT" else df_cmg
df = df.tail(days_to_show).copy()

# ======= 4. Tabs phân tích =======
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "TỔNG QUAN", 
    "BIẾN ĐỘNG GIÁ", 
    "BOLLINGER BANDS", 
    "MACD & RSI", 
    "DỮ LIỆU"
])

with tab1:
    # === Tab tổng quan ===
    st.subheader(f"Tổng quan {selected_stock}")
    
    # Thông số chính
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá hiện tại", f"{df['Closing Price'].iloc[-1]:,.2f}")
    with col2:
        st.metric("Thay đổi 1 ngày", 
                 f"{df['Daily_Change'].iloc[-1]:,.2f}", 
                 f"{df['Daily_Change_Pct'].iloc[-1]:.2f}%")
    with col3:
        st.metric("Tổng biến động", 
                 f"{df['Cumulative_Change'].iloc[-1]:,.2f}", 
                 f"{df['Cumulative_Change_Pct'].iloc[-1]*100:.2f}%")
    
    # Biểu đồ tổng hợp
    fig_overview = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Giá và đường trung bình
    fig_overview.add_trace(
        go.Scatter(x=df['Date'], y=df['Closing Price'], name='Giá đóng cửa', line=dict(color='#0E6994')),
        row=1, col=1
    )
    fig_overview.add_trace(
        go.Scatter(x=df['Date'], y=df['SMA_14'], name='SMA 14', line=dict(color='#FD6200')),
        row=1, col=1
    )
    
    # Khối lượng
    fig_overview.add_trace(
        go.Bar(x=df['Date'], y=df['Total Volume'], name='Khối lượng', marker_color='#7EC8E3'),
        row=2, col=1
    )
    
    fig_overview.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig_overview, use_container_width=True)

with tab2:
   
    st.subheader(f"Biến động giá {selected_stock} ({days_to_show} ngày gần nhất)")
    # === Tab biến động giá ===
    
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

with tab3:
    # === Tab Bollinger Bands ===
    st.subheader("Bollinger Bands với Cảnh báo Breakout")
    
    fig_bb = go.Figure()
    
    # Dải Bollinger
    fig_bb.add_trace(go.Scatter(
        x=df['Date'], y=df['BB_upper'], line=dict(color='rgba(0,0,0,0)'), showlegend=False
    ))
    fig_bb.add_trace(go.Scatter(
        x=df['Date'], y=df['BB_lower'], fill='tonexty',
        fillcolor='rgba(132,208,208,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Bollinger Band'
    ))
    
    # Giá và đường giữa
    fig_bb.add_trace(go.Scatter(
        x=df['Date'], y=df['Closing Price'], name='Giá đóng cửa', line=dict(color='#0E6994', width=2)
    ))
    fig_bb.add_trace(go.Scatter(
        x=df['Date'], y=df['BB_middle'], name='BB Middle', line=dict(color='#FD6200', width=2, dash='dot')
    ))
    
    # Điểm breakout
    breakout_up = df[df['Closing Price'] > df['BB_upper']]
    breakout_down = df[df['Closing Price'] < df['BB_lower']]
    
    if not breakout_up.empty:
        fig_bb.add_trace(go.Scatter(
            x=breakout_up['Date'], y=breakout_up['Closing Price'], mode='markers',
            name='Breakout ↑', marker=dict(color='green', symbol='triangle-up', size=10)
        ))
    
    if not breakout_down.empty:
        fig_bb.add_trace(go.Scatter(
            x=breakout_down['Date'], y=breakout_down['Closing Price'], mode='markers',
            name='Breakout ↓', marker=dict(color='red', symbol='triangle-down', size=10)
        ))
    
    fig_bb.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig_bb, use_container_width=True)

    with st.expander("📌 Giải thích tín hiệu"):
        st.markdown("""
        - **Dải Bollinger**: Đo lường biến động giá
        - **Giá chạm dải trên**: Có thể quá mua (overbought)
        - **Giá chạm dải dưới**: Có thể quá bán (oversold)
        - **Bóp dải**: Chuẩn bị có biến động mạnh
        """)


with tab4:
    # === Tab MACD & RSI ===
    st.subheader("MACD và RSI")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='#0E6994')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Signal_Line'], name='Signal', line=dict(color='#FD6200', dash='dot')
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Histogram'], name='Histogram',
        marker_color=['green' if x >=0 else 'red' for x in df['Histogram']]
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['RSI_14'], name='RSI 14', line=dict(color='#A05195')
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=700, hovermode="x unified", yaxis2=dict(range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📌 Giải thích tín hiệu"):
        st.markdown("""
        - **MACD > Signal**: Tín hiệu tăng giá
        - **MACD < Signal**: Tín hiệu giảm giá
        - **Histogram tăng**: Động lực tăng đang mạnh lên
        - **Histogram giảm**: Động lực tăng đang yếu đi
        """)

    with st.expander("📌 Giải thích tín hiệu"):
        st.markdown("""
        - **RSI > 70**: Quá mua, có thể điều chỉnh giảm
        - **RSI < 30**: Quá bán, có thể phục hồi
        - **Phân kỳ RSI**: Cảnh báo đảo chiều tiềm năng
        """)

with tab5:
    # === Tab dữ liệu ===
    st.subheader("Dữ liệu Chi tiết")
    
    display_cols = [
        'Date', 'Closing Price', 'Price Change', 'Price Change %', 
        'SMA_14', 'EMA_12', 'EMA_26', 'MACD', 'RSI_14',
        'BB_upper', 'BB_middle', 'BB_lower'
    ]
    
    st.dataframe(
        df[display_cols].sort_values('Date', ascending=False)
        .style.format({
            'Closing Price': '{:,.2f}',
            'Price Change': '{:,.2f}',
            'Price Change %': '{:.2f}%',
            'SMA_14': '{:,.2f}',
            'EMA_12': '{:,.2f}',
            'EMA_26': '{:,.2f}',
            'MACD': '{:,.2f}',
            'RSI_14': '{:.2f}',
            'BB_upper': '{:,.2f}',
            'BB_middle': '{:,.2f}',
            'BB_lower': '{:,.2f}'
        })
        .background_gradient(subset=['Price Change', 'Price Change %'], cmap='RdYlGn'),
        use_container_width=True,
        height=500
    )
