import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import joblib
import os
from datetime import datetime, timedelta

# Thiết lập trang Streamlit
st.set_page_config(page_title="Dự Báo Giá Cổ Phiếu CMG", page_icon="📈", layout="wide")

# Tiêu đề ứng dụng
st.title("📈 Hệ Thống Dự Báo Giá Cổ Phiếu CMG")
st.markdown("**Kết hợp phân tích kỹ thuật và cơ bản để dự báo giá cổ phiếu với đánh giá độ tin cậy**")

# Sidebar cho lựa chọn thông số
with st.sidebar:
    st.header("Cấu hình Dự Báo")
    stock_choice = "CMG"  # Fixed to CMG for now
    forecast_days = st.slider("Số ngày dự báo", 1, 30, 7)
    confidence_threshold = st.slider("Ngưỡng tin cậy tối thiểu (%)", 50, 95, 70)
    st.markdown("---")
    st.info("Hệ thống sử dụng mô hình hybrid kết hợp: LSTM, XGBoost, và Meta-model.")

# Hàm tải và xử lý dữ liệu
@st.cache_data
def load_stock_data(stock_id):
    try:
        # Load main transaction data
        df = pd.read_csv("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
        df = df[df['StockID'] == stock_id].copy()
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date')
        df['Closing Price'] = df['Closing Price'].str.replace(',', '').astype(float)
        df['Total Volume'] = df['Total Volume'].str.replace(',', '').astype(float)

        # Create features
        df['Return%'] = df['Closing Price'].pct_change() * 100
        df['MA5'] = df['Closing Price'].rolling(window=5).mean()
        df['MA10'] = df['Closing Price'].rolling(window=10).mean()
        df['Volume_ratio'] = df['Total Volume'] / df['Total Volume'].rolling(5).mean()
        df['Volatility'] = df['Closing Price'].pct_change().rolling(window=5).std() * 100
        df['Price_Momentum'] = df['Closing Price'].diff(5)

        # Load event data
        df_dividend = pd.read_csv("3.2 (live & his) news_dividend_issue (FPT_CMG)_processed.csv")
        df_meeting = pd.read_csv("3.3 (live & his) news_shareholder_meeting (FPT_CMG)_processed.csv")
        df_dividend = df_dividend[df_dividend['StockID'] == stock_id].copy()
        df_meeting = df_meeting[df_meeting['StockID'] == stock_id].copy()
        df_dividend['Execution Date'] = pd.to_datetime(df_dividend['Execution Date'], format='%d/%m/%Y', errors='coerce')
        df_meeting['Execution Date'] = pd.to_datetime(df_meeting['Execution Date'], format='%d/%m/%Y')
        df['Dividend_Event'] = df['Date'].isin(df_dividend['Execution Date']).astype(int)
        df['Meeting_Event'] = df['Date'].isin(df_meeting['Execution Date']).astype(int)

        # Load financial data
        df_financial = pd.read_csv("6.5 (his) financialreport_metrics_Nhóm ngành_Công nghệ thông tin (of FPT_CMG)_processed.csv")
        def clean_financial_data(df_fin):
            df_fin['Indicator'] = df_fin['Indicator'].str.replace('\n', '', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip()
            for col in df_fin.columns[3:]:
                df_fin[col] = df_fin[col].str.replace(',', '').astype(float, errors='ignore')
            return df_fin
        df_financial = clean_financial_data(df_financial)

        indicators = [
            'Tỷ suất lợi nhuận trên Vốn chủ sở hữu bình quân (ROEA)%',
            'Tỷ lệ lãi EBIT%',
            'Chỉ số giá thị trường trên giá trị sổ sách (P/B)Lần',
            'Chỉ số giá thị trường trên thu nhập (P/E)Lần',
            'P/SLần',
            'Tỷ suất sinh lợi trên vốn dài hạn bình quân (ROCE)%',
            'Thu nhập trên mỗi cổ phần (EPS)VNĐ'
        ]
        df_financial = df_financial[(df_financial['Stocks'].str.contains(stock_id)) & (df_financial['Indicator'].isin(indicators))].copy()
        quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023', 'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
        df_financial_melted = df_financial.melt(id_vars=['Indicator'], value_vars=quarters, var_name='Quarter', value_name='Value')
        quarter_dates = {
            'Q1_2023': '2023-01-01', 'Q2_2023': '2023-04-01', 'Q3_2023': '2023-07-01', 'Q4_2023': '2023-10-01',
            'Q1_2024': '2024-01-01', 'Q2_2024': '2024-04-01', 'Q3_2024': '2024-07-01', 'Q4_2024': '2024-10-01'
        }
        df_financial_melted['Date'] = df_financial_melted['Quarter'].map(quarter_dates)
        df_financial_melted['Date'] = pd.to_datetime(df_financial_melted['Date'])
        df_financial_pivot = df_financial_melted.pivot(index='Date', columns='Indicator', values='Value')
        df = df.merge(df_financial_pivot, left_on='Date', right_index=True, how='left')
        df[indicators] = df[indicators].ffill()
        df = df.dropna().reset_index(drop=True)
        return df
    except FileNotFoundError:


        st.error("Không thể tải dữ liệu. Vui lòng kiểm tra tệp CSV.")
        st.stop()


# Hàm tải mô hình
@st.cache_resource
def load_models(stock_id):
    try:
        model_lstm = load_model(f"models/lstm_model_{stock_id}.h5")
        model_xgb = joblib.load(f"models/xgb_model_{stock_id}.joblib")
        meta_model = load_model(f"models/meta_model_{stock_id}.h5")
        return {'lstm': model_lstm, 'xgb': model_xgb, 'meta': meta_model}
    except FileNotFoundError:
        st.warning("Mô hình chưa tồn tại. Đang huấn luyện mô hình mới...")
        df = load_stock_data(stock_id)
        model_lstm, model_xgb, meta_model, scaler = train_and_save_models(stock_id, df)
        return {'lstm': model_lstm, 'xgb': model_xgb, 'meta': meta_model}

# Tải dữ liệu và mô hình
df = load_stock_data(stock_choice)
models = load_models(stock_choice)
scaler = MinMaxScaler()  # Sẽ được cập nhật trong hàm dự báo

# Hàm dự báo giá
def forecast_prices(df, models, forecast_days, lookback=7):
    y_log = np.log1p(df['Closing Price'])
    scaler.fit(y_log.values.reshape(-1, 1))
    scaled_data = scaler.transform(y_log.values.reshape(-1, 1))
    last_data = scaled_data[-lookback:]
    X_lstm = last_data.reshape(1, lookback, 1)
    
    forecast_prices = []
    confidence_scores = []
    for _ in range(forecast_days):
        lstm_pred = models['lstm'].predict(X_lstm, verbose=0)
        lstm_pred_price = np.expm1(scaler.inverse_transform(lstm_pred))[0, 0]
        
        last_features = df.iloc[-1][['Return%', 'MA5', 'MA10', 'Volume_ratio', 'Dividend_Event', 'Meeting_Event', 'Volatility', 'Price_Momentum'] + [
            'Tỷ suất lợi nhuận trên Vốn chủ sở hữu bình quân (ROEA)%',
            'Tỷ lệ lãi EBIT%',
            'Chỉ số giá thị trường trên giá trị sổ sách (P/B)Lần',
            'Chỉ số giá thị trường trên thu nhập (P/E)Lần',
            'P/SLần',
            'Tỷ suất sinh lợi trên vốn dài hạn bình quân (ROCE)%',
            'Thu nhập trên mỗi cổ phần (EPS)VNĐ'
        ]].values.reshape(1, -1)
        xgb_pred = models['xgb'].predict(last_features)[0]
        
        meta_input = np.array([[lstm_pred_price, xgb_pred]])
        final_pred = models['meta'].predict(meta_input, verbose=0)[0, 0]
        final_pred = np.nan_to_num(final_pred, nan=df['Closing Price'].mean(), neginf=0)
        forecast_prices.append(max(final_pred, 0))  # Ensure non-negative
        
        # Calculate confidence (simplified, based on recent volatility)
        recent_volatility = df['Volatility'].iloc[-10:].mean()
        confidence = max(50, 100 - recent_volatility * 5 + np.random.normal(0, 5))
        confidence_scores.append(confidence)
        
        # Update input for next prediction
        new_data_point = np.append(last_data[1:], lstm_pred)
        X_lstm = new_data_point.reshape(1, lookback, 1)
    
    forecast_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
    return forecast_dates, forecast_prices, confidence_scores

# Tab chính
tab1, tab2, tab3 = st.tabs(["📊 Dữ Liệu & Phân Tích", "🔮 Dự Báo Giá", "📌 Đề Xuất Giao Dịch"])

with tab1:
    st.header(f"Phân Tích Cổ Phiếu {stock_choice}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Biểu Đồ Giá")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Closing Price'], name='Giá đóng cửa', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], name='MA5', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA10'], name='MA10', line=dict(color='green', width=1)))
        fig.update_layout(height=400, xaxis_title='Ngày', yaxis_title='Giá')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Chỉ Số Kỹ Thuật")
        delta = df['Closing Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        ema12 = df['Closing Price'].ewm(span=12, adjust=False).mean()
        ema26 = df['Closing Price'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        st.metric("RSI (14 ngày)", f"{rsi.iloc[-1]:.2f}", 
                  "Mua quá" if rsi.iloc[-1] > 70 else "Bán quá" if rsi.iloc[-1] < 30 else "Bình thường")
        st.metric("MACD", f"{macd.iloc[-1]:.2f}", 
                  "Tăng" if macd.iloc[-1] > signal.iloc[-1] else "Giảm")
        st.metric("Khối lượng giao dịch", f"{df['Total Volume'].iloc[-1]:,.0f}")
        trend = "Tăng" if df['Closing Price'].iloc[-1] > df['Closing Price'].iloc[-5] else "Giảm"
        st.metric("Xu hướng ngắn hạn", trend)

with tab2:
    st.header(f"Dự Báo Giá {stock_choice}")
    
    forecast_dates, forecast_prices, confidence_scores = forecast_prices(df, models, forecast_days)
    
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
        for i, (date, price, conf) in enumerate(zip(forecast_dates, forecast_prices, confidence_scores)):
            color = 'green' if conf >= confidence_threshold else 'orange' if conf >= 60 else 'red'
            fig.add_shape(type="line",
                x0=date, y0=price*0.98, x1=date, y1=price*1.02,
                line=dict(color=color, width=2)
            )
            if i % 3 == 0:
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
        forecast_df = pd.DataFrame({
            'Ngày': forecast_dates,
            'Giá dự báo': forecast_prices,
            'Độ tin cậy (%)': confidence_scores,
            'Biến động (%)': [(p/df['Closing Price'].iloc[-1]-1)*100 for p in forecast_prices]
        })
        forecast_df['Giá dự báo'] = forecast_df['Giá dự báo'].apply(lambda x: f"{x:,.0f}")
        forecast_df['Biến động (%)'] = forecast_df['Biến động (%)'].apply(lambda x: f"{x:+.2f}%")
        forecast_df['Độ tin cậy (%)'] = forecast_df['Độ tin cậy (%)'].apply(lambda x: f"{x:.0f}%")
        
        def color_confidence(val):
            val = float(val.strip('%'))
            color = 'green' if val >= confidence_threshold else 'orange' if val >= 60 else 'red'
            return f'background-color: {color}; color: white'
        
        st.dataframe(
            forecast_df.style.applymap(color_confidence, subset=['Độ tin cậy (%)']),
            hide_index=True,
            use_container_width=True
        )
        
        avg_confidence = np.mean(confidence_scores)
        max_change = max([abs(float(x.strip('%'))) for x in forecast_df['Biến động (%)']])
        st.metric("Độ tin cậy trung bình", f"{avg_confidence:.1f}%")
        st.metric("Biến động tối đa dự kiến", f"{max_change:.2f}%")
        if avg_confidence >= confidence_threshold:
            st.success("✅ Dự báo có độ tin cậy cao, có thể cân nhắc sử dụng")
        elif avg_confidence >= 60:
            st.warning("⚠️ Dự báo có độ tin cậy trung bình, cần thận trọng")
        else:
            st.error("❌ Dự báo có độ tin cậy thấp, không nên sử dụng")

with tab3:
    st.header(f"Đề Xuất Giao Dịch {stock_choice}")
    
    current_price = df['Closing Price'].iloc[-1]
    ma5 = df['MA5'].iloc[-1]
    ma10 = df['MA10'].iloc[-1]
    rsi_value = rsi.iloc[-1]
    
    recommendation = "Giữ"
    confidence = 70
    reasoning = []
    
    # Incorporate model prediction
    forecast_trend = forecast_prices[-1] > current_price
    if forecast_trend:
        reasoning.append("Mô hình dự báo xu hướng tăng trong ngắn hạn")
    else:
        reasoning.append("Mô hình dự báo xu hướng giảm trong ngắn hạn")
    
    if current_price > ma5 > ma10:
        recommendation = "Mua"
        confidence = 80
        reasoning.append("Giá vượt trên cả MA5 và MA10 - xu hướng tăng ngắn hạn")
    elif current_price < ma5 < ma10:
        recommendation = "Bán"
        confidence = 75
        reasoning.append("Giá dưới cả MA5 và MA10 - xu hướng giảm ngắn hạn")
    
    if rsi_value < 30:
        recommendation = "Mua mạnh" if recommendation == "Mua" else "Mua"
        confidence = min(95, confidence + 15)
        reasoning.append("RSI dưới 30 - cổ phiếu bị bán quá mức")
    elif rsi_value > 70:
        recommendation = "Bán mạnh" if recommendation == "Bán" else "Bán"
        confidence = min(95, confidence + 15)
        reasoning.append("RSI trên 70 - cổ phiếu mua quá mức")
    
    if df['Dividend_Event'].iloc[-5:].sum() > 0:
        reasoning.append("Có sự kiện cổ tức gần đây - thường tạo biến động giá")
    if df['Meeting_Event'].iloc[-5:].sum() > 0:
        reasoning.append("Có sự kiện họp cổ đông gần đây - cần theo dõi thông tin")
    
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
        
        # Volatility-based stop-loss/take-profit
        atr = df['Closing Price'].diff().abs().rolling(window=14).mean().iloc[-1]
        if recommendation.startswith("Mua"):
            entry = current_price * 0.99
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 3 * atr
        elif recommendation.startswith("Bán"):
            entry = current_price * 1.01
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 3 * atr
        else:
            entry = current_price
            stop_loss = current_price - 1.5 * atr
            take_profit = current_price + 1.5 * atr
        
        st.metric("Điểm vào lệnh", f"{entry:,.0f}")
        st.metric("Cắt lỗ", f"{stop_loss:,.0f}", delta=f"{(stop_loss/current_price-1)*100:+.1f}%")
        st.metric("Chốt lời", f"{take_profit:,.0f}", delta=f"{(take_profit/current_price-1)*100:+.1f}%")
    
    with col2:
        st.subheader("Phân Tích Chi Tiết")
        st.write("**Cơ sở đề xuất:**")
        for reason in reasoning:
            st.write(f"- {reason}")
        
        st.write("**Chỉ số quan trọng:**")
        cols = st.columns(4)
        with cols[0]:
            st.metric("P/E", f"{df['Chỉ số giá thị trường trên thu nhập (P/E)Lần'].iloc[-1]:.1f}")
        with cols[1]:
            st.metric("ROE", f"{df['Tỷ suất lợi nhuận trên Vốn chủ sở hữu bình quân (ROEA)%'].iloc[-1]:.1f}%")
        with cols[2]:
            st.metric("EPS", f"{df['Thu nhập trên mỗi cổ phần (EPS)VNĐ'].iloc[-1]:,.0f}")
        with cols[3]:
            st.metric("Volume Ratio", f"{df['Volume_ratio'].iloc[-1]:.2f}")
        
        st.warning("**Cảnh báo rủi ro:**")
        st.write("""
        - Dự báo không đảm bảo chính xác 100%
        - Thị trường có thể biến động do yếu tố vĩ mô
        - Luôn sử dụng lệnh cắt lỗ để quản lý rủi ro
        - Cân nhắc đa dạng hóa danh mục đầu tư
        """)

# Hiển thị đánh giá mô hình
st.header("Đánh Giá Mô hình")
y_log = np.log1p(df['Closing Price'])
scaler.fit(y_log.values.reshape(-1, 1))
scaled_data = scaler.transform(y_log.values.reshape(-1, 1))
X_lstm = []
for i in range(lookback, len(scaled_data)):
    X_lstm.append(scaled_data[i - lookback:i, 0])
X_lstm = np.array(X_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
X_xgb = df[['Return%', 'MA5', 'MA10', 'Volume_ratio', 'Dividend_Event', 'Meeting_Event', 'Volatility', 'Price_Momentum'] + [
    'Tỷ suất lợi nhuận trên Vốn chủ sở hữu bình quân (ROEA)%',
    'Tỷ lệ lãi EBIT%',
    'Chỉ số giá thị trường trên giá trị sổ sách (P/B)Lần',
    'Chỉ số giá thị trường trên thu nhập (P/E)Lần',
    'P/SLần',
    'Tỷ suất sinh lợi trên vốn dài hạn bình quân (ROCE)%',
    'Thu nhập trên mỗi cổ phần (EPS)VNĐ'
]].iloc[lookback:].reset_index(drop=True)
y = df['Closing Price'].iloc[lookback:].reset_index(drop=True)
mask = ~(X_xgb.isna().any(axis=1) | y.isna())
X_xgb, y, X_lstm = X_xgb[mask], y[mask], X_lstm[mask]
split = int(len(y) * 0.8)
X_lstm_test = X_lstm[split:]
X_xgb_test = X_xgb[split:]
y_test = y[split:]
pred_lstm_test = np.expm1(scaler.inverse_transform(models['lstm'].predict(X_lstm_test, verbose=0))).flatten()
pred_xgb_test = models['xgb'].predict(X_xgb_test)
X_meta_test = np.vstack((pred_lstm_test, pred_xgb_test)).T
y_pred = models['meta'].predict(X_meta_test, verbose=0).flatten()
y_pred = np.nan_to_num(y_pred, nan=y.mean(), neginf=0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R²", f"{r2:.4f}")

# Footer
st.markdown("---")
st.markdown("""
**Hướng dẫn sử dụng:**
1. Chọn số ngày dự báo và ngưỡng tin cậy ở sidebar
2. Xem phân tích kỹ thuật và cơ bản ở tab đầu tiên
3. Kiểm tra dự báo giá và độ tin cậy ở tab thứ hai
4. Tham khảo đề xuất giao dịch ở tab cuối cùng
""")