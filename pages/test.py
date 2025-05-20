import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib
import os
from datetime import datetime, timedelta

# Thiết lập trang Streamlit
st.set_page_config(
    page_title="Dự Báo Giá Cổ Phiếu FPT & CMG",
    page_icon="📈",
    layout="wide"
)

# Tiêu đề ứng dụng
st.title("📈 Hệ Thống Dự Báo Giá Cổ Phiếu FPT & CMG")
st.markdown("""
**Kết hợp phân tích kỹ thuật và cơ bản để dự báo giá cổ phiếu với đánh giá độ tin cậy**
""")

# Sidebar cho lựa chọn cổ phiếu và thông số
with st.sidebar:
    st.header("Cấu hình Dự Báo")
    stock_choice = st.selectbox("Chọn cổ phiếu", ["FPT", "CMG"])
    forecast_days = st.slider("Số ngày dự báo", 1, 30, 7)
    confidence_threshold = st.slider("Ngưỡng tin cậy tối thiểu (%)", 50, 95, 70)
    st.markdown("---")
    st.info("""
    Hệ thống sử dụng mô hình hybrid kết hợp:
    - LSTM cho phân tích chuỗi thời gian
    - XGBoost cho phân tích đặc trưng
    - Meta-model để kết hợp kết quả
    """)

# Hàm tải dữ liệu (giả lập - thay bằng dữ liệu thực tế của bạn)
@st.cache_data
def load_stock_data(stock_id):
    # Đây là phần giả lập - thay bằng code tải dữ liệu thực tế của bạn
    # Từ code gốc của bạn, bạn cần thay thế phần này với các file CSV thực tế
    
    # Tạo dữ liệu giả lập cho demo
    date_range = pd.date_range(end=datetime.today(), periods=365)
    prices = np.cumsum(np.random.normal(0.1, 2, 365)) + 100
    
    df = pd.DataFrame({
        'Date': date_range,
        'Closing Price': prices,
        'Total Volume': np.random.randint(100000, 500000, 365),

        'Volume_ratio': np.random.uniform(0.8, 1.2, 365),
        'Volatility': np.random.uniform(1, 5, 365),
        'Price_Momentum': np.random.normal(0, 2, 365),
        'Dividend_Event': np.random.choice([0, 1], 365, p=[0.95, 0.05]),
        'Meeting_Event': np.random.choice([0, 1], 365, p=[0.9, 0.1]),
        'ROE': np.random.uniform(10, 20, 365),
        'P/E': np.random.uniform(15, 25, 365),
        'EPS': np.random.uniform(5000, 8000, 365)
    })
    
    df['Return%'] = df['Closing Price'].pct_change() * 100
    df = df.fillna(0)
    
    return df

# Hàm tải mô hình
@st.cache_resource
def load_models(stock_id):
    # Trong thực tế, bạn cần thay bằng đường dẫn đến các model đã train
    # Đây chỉ là phần giả lập
    
    class DummyModel:
        def predict(self, X):
            return np.random.normal(0, 1, len(X))
    
    return {
        'lstm': DummyModel(),
        'xgb': DummyModel(),
        'meta': DummyModel()
    }

# Tải dữ liệu và mô hình
df = load_stock_data(stock_choice)
models = load_models(stock_choice)

# Tab chính
tab1, tab2, tab3 = st.tabs(["📊 Dữ Liệu & Phân Tích", "🔮 Dự Báo Giá", "📌 Đề Xuất Giao Dịch"])

with tab1:
    st.header(f"Phân Tích Cổ Phiếu {stock_choice}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Biểu Đồ Giá")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Closing Price'], name='Giá đóng cửa', line=dict(color='blue')))

        fig.update_layout(height=400, xaxis_title='Ngày', yaxis_title='Giá')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Chỉ Số Kỹ Thuật")
        
        # Tính RSI
        delta = df['Closing Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Tính MACD
        ema12 = df['Closing Price'].ewm(span=12, adjust=False).mean()
        ema26 = df['Closing Price'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Hiển thị các chỉ số
        st.metric("RSI (14 ngày)", f"{rsi.iloc[-1]:.2f}", 
                  "Mua quá" if rsi.iloc[-1] > 70 else "Bán quá" if rsi.iloc[-1] < 30 else "Bình thường")
        st.metric("MACD", f"{macd.iloc[-1]:.2f}", 
                  "Tăng" if macd.iloc[-1] > signal.iloc[-1] else "Giảm")
        st.metric("Khối lượng giao dịch", f"{df['Total Volume'].iloc[-1]:,.0f}")
        
        # Phân tích xu hướng
        trend = "Tăng" if df['Closing Price'].iloc[-1] > df['Closing Price'].iloc[-5] else "Giảm"
        st.metric("Xu hướng ngắn hạn", trend)

with tab2:
    st.header(f"Dự Báo Giá {stock_choice}")
    
    # Chuẩn bị dữ liệu cho dự báo (giả lập)
    lookback = 7
    last_data = df.iloc[-lookback:][['Closing Price']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(last_data)
    
    # Tạo dự báo (giả lập)
    forecast_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
    forecast_prices = []
    confidence_scores = []
    
    for i in range(forecast_days):
        # Trong thực tế, bạn sẽ sử dụng model.predict()
        pred = df['Closing Price'].iloc[-1] * (1 + np.random.normal(0.001, 0.02))
        forecast_prices.append(pred)
        
        # Tính độ tin cậy giả lập (dựa trên độ biến động gần đây)
        recent_volatility = df['Volatility'].iloc[-10:].mean()
        confidence = max(50, 100 - recent_volatility * 5 + np.random.normal(10, 5))
        confidence_scores.append(confidence)
    
    # Hiển thị kết quả dự báo
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Biểu Đồ Dự Báo")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'].iloc[-30:], 
            y=df['Closing Price'].iloc[-30:], 
            name='Giá lịch sử',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_prices,
            name='Dự báo',
            line=dict(color='red', dash='dot')
        ))
        
        # Thêm vùng độ tin cậy
        for i, (date, price, conf) in enumerate(zip(forecast_dates, forecast_prices, confidence_scores)):
            color = 'green' if conf >= confidence_threshold else 'orange' if conf >= 60 else 'red'
            fig.add_shape(type="line",
                x0=date, y0=price*0.98, x1=date, y1=price*1.02,
                line=dict(color=color, width=2)
            )
            if i % 3 == 0:  # Hiển thị nhãn cho một số ngày để tránh rối
                fig.add_annotation(x=date, y=price*1.03,
                    text=f"{conf:.0f}%",
                    showarrow=False,
                    font=dict(size=10, color=color)
                )
        
        fig.update_layout(
            height=500,
            title=f"Dự báo giá {stock_choice} trong {forecast_days} ngày tới",
            xaxis_title="Ngày",
            yaxis_title="Giá",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Chi Tiết Dự Báo")
        
        # Tạo bảng dự báo
        forecast_df = pd.DataFrame({
            'Ngày': forecast_dates,
            'Giá dự báo': forecast_prices,
            'Độ tin cậy (%)': confidence_scores,
            'Biến động (%)': [(p/df['Closing Price'].iloc[-1]-1)*100 for p in forecast_prices]
        })
        
        # Định dạng bảng
        forecast_df['Giá dự báo'] = forecast_df['Giá dự báo'].apply(lambda x: f"{x:,.0f}")
        forecast_df['Biến động (%)'] = forecast_df['Biến động (%)'].apply(lambda x: f"{x:+.2f}%")
        forecast_df['Độ tin cậy (%)'] = forecast_df['Độ tin cậy (%)'].apply(lambda x: f"{x:.0f}%")
        
        # Hiển thị bảng với màu sắc theo độ tin cậy
        def color_confidence(val):
            val = float(val.strip('%'))
            color = 'green' if val >= confidence_threshold else 'orange' if val >= 60 else 'red'
            return f'background-color: {color}; color: white'
        
        st.dataframe(
            forecast_df.style.applymap(color_confidence, subset=['Độ tin cậy (%)']),
            hide_index=True,
            use_container_width=True
        )
        
        # Thống kê dự báo
        avg_confidence = np.mean(confidence_scores)
        max_change = max([abs(float(x.strip('%'))) for x in forecast_df['Biến động (%)']])
        
        st.metric("Độ tin cậy trung bình", f"{avg_confidence:.1f}%")
        st.metric("Biến động tối đa dự kiến", f"{max_change:.2f}%")
        
        # Đánh giá tổng quan
        if avg_confidence >= confidence_threshold:
            st.success("✅ Dự báo có độ tin cậy cao, có thể cân nhắc sử dụng")
        elif avg_confidence >= 60:
            st.warning("⚠️ Dự báo có độ tin cậy trung bình, cần thận trọng")
        else:
            st.error("❌ Dự báo có độ tin cậy thấp, không nên sử dụng")

with tab3:
    st.header(f"Đề Xuất Giao Dịch {stock_choice}")
    
    # Phân tích kỹ thuật để đưa ra đề xuất
    current_price = df['Closing Price'].iloc[-1]

    rsi_value = rsi.iloc[-1] if 'rsi' in locals() else 50  # Sử dụng RSI đã tính ở tab1
    
    # Tạo đề xuất
    recommendation = "Giữ"
    confidence = 70
    reasoning = []
    

    
    if rsi_value < 30:
        recommendation = "Mua mạnh" if recommendation == "Mua" else "Mua"
        confidence = min(95, confidence + 15)
        reasoning.append("RSI dưới 30 - cổ phiếu bị bán quá mức")
    elif rsi_value > 70:
        recommendation = "Bán mạnh" if recommendation == "Bán" else "Bán"
        confidence = min(95, confidence + 15)
        reasoning.append("RSI trên 70 - cổ phiếu mua quá mức")
    
    # Xem xét các sự kiện công ty
    if df['Dividend_Event'].iloc[-5:].sum() > 0:
        reasoning.append("Có sự kiện cổ tức gần đây - thường tạo biến động giá")
    
    if df['Meeting_Event'].iloc[-5:].sum() > 0:
        reasoning.append("Có sự kiện họp cổ đông gần đây - cần theo dõi thông tin")
    
    # Hiển thị đề xuất
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Khuyến Nghị")
        
        if recommendation.startswith("Mua"):
            st.success(f"### {recommendation}")
        elif recommendation.startswith("Bán"):
            st.error(f"### {recommendation}")
        else:
            st.info(f"### {recommendation}")
        
        st.metric("Độ tin cậy", f"{confidence}%")
        st.metric("Giá hiện tại", f"{current_price:,.0f}")
        
        # Điểm vào lệnh và cắt lỗ đề xuất
        if recommendation.startswith("Mua"):
            entry = current_price * 0.99
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.08
        elif recommendation.startswith("Bán"):
            entry = current_price * 1.01
            stop_loss = current_price * 1.05
            take_profit = current_price * 0.92
        else:
            entry = current_price
            stop_loss = current_price * 0.97
            take_profit = current_price * 1.03
        
        st.metric("Điểm vào lệnh", f"{entry:,.0f}")
        st.metric("Cắt lỗ", f"{stop_loss:,.0f}", delta=f"{(stop_loss/current_price-1)*100:+.1f}%")
        st.metric("Chốt lời", f"{take_profit:,.0f}", delta=f"{(take_profit/current_price-1)*100:+.1f}%")
    
    with col2:
        st.subheader("Phân Tích Chi Tiết")
        
        # Hiển thị lý do
        st.write("**Cơ sở đề xuất:**")
        for reason in reasoning:
            st.write(f"- {reason}")
        
        # Hiển thị các chỉ số quan trọng
        st.write("**Chỉ số quan trọng:**")
        cols = st.columns(4)
        with cols[0]:
            st.metric("P/E", f"{df['P/E'].iloc[-1]:.1f}")
        with cols[1]:
            st.metric("ROE", f"{df['ROE'].iloc[-1]:.1f}%")
        with cols[2]:
            st.metric("EPS", f"{df['EPS'].iloc[-1]:,.0f}")
        with cols[3]:
            st.metric("Volume Ratio", f"{df['Volume_ratio'].iloc[-1]:.2f}")
        
        # Cảnh báo rủi ro
        st.warning("**Cảnh báo rủi ro:**")
        st.write("""
        - Dự báo không đảm bảo chính xác 100%
        - Thị trường có thể biến động do yếu tố vĩ mô
        - Luôn sử dụng lệnh cắt lỗ để quản lý rủi ro
        - Cân nhắc đa dạng hóa danh mục đầu tư
        """)

# Footer
st.markdown("---")
st.markdown("""
**Hướng dẫn sử dụng:**
1. Chọn cổ phiếu và số ngày dự báo ở sidebar
2. Xem phân tích kỹ thuật và cơ bản ở tab đầu tiên
3. Kiểm tra dự báo giá và độ tin cậy ở tab thứ hai
4. Tham khảo đề xuất giao dịch ở tab cuối cùng
""")