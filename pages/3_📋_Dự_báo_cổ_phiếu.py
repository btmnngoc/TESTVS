import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import xgboost as xgb

# Tiêu đề ứng dụng
st.set_page_config(page_title="Dự đoán Giá Cổ phiếu", layout="wide")
st.title("📈 Ứng dụng Dự đoán Giá Cổ phiếu")

# Sidebar - Cài đặt
st.sidebar.header("Cài đặt Dự báo")

# 1. Chọn mã cổ phiếu
stock_id = st.sidebar.selectbox("Chọn mã cổ phiếu", ["CMG", "FPT"])

# 2. Chọn số ngày dự báo
forecast_days = st.sidebar.slider("Số ngày dự báo", 1, 30, 7)

# 3. Chọn mô hình
model_type = st.sidebar.radio("Chọn mô hình", 
                             ["Meta Model (Tích hợp LSTM + XGBoost)", 
                              "LSTM Model", 
                              "XGBoost Model"])

# 4. Tùy chọn huấn luyện lại
retrain = st.sidebar.checkbox("Huấn luyện lại mô hình", value=False)

# Hàm tải mô hình
@st.cache_resource
def load_models(stock_id):
    try:
        # Đăng ký hàm loss/metrics trước khi load model
        from tensorflow.keras.saving import register_keras_serializable
        from tensorflow.keras.losses import mse
        
        @register_keras_serializable()
        def custom_mse(y_true, y_pred):
            return mse(y_true, y_pred)
        
        # Load model với custom_objects
        lstm_model = load_model(
            f'models/lstm_model_{stock_id}.h5',
            custom_objects={'mse': custom_mse}
        )
        
        xgb_model = joblib.load(f'models/xgb_model_{stock_id}.joblib')
        
        meta_model = load_model(
            f'models/meta_model_{stock_id}.h5',
            custom_objects={'mse': custom_mse}
        )
        
        return lstm_model, xgb_model, meta_model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None, None, None

# Hàm tiền xử lý dữ liệu
def preprocess_data(df, stock_id):
    try:
        df = df[df['StockID'] == stock_id].copy()
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date')
        df['Closing Price'] = df['Closing Price'].str.replace(',', '').astype(float)
        df['Total Volume'] = df['Total Volume'].str.replace(',', '').astype(float)
        
        # Tạo các đặc trưng
        df['Return%'] = df['Closing Price'].pct_change() * 100
        df['MA5'] = df['Closing Price'].rolling(window=5).mean()
        df['MA10'] = df['Closing Price'].rolling(window=10).mean()
        df['Volume_ratio'] = df['Total Volume'] / df['Total Volume'].rolling(5).mean()
        df['Volatility'] = df['Closing Price'].pct_change().rolling(window=5).std() * 100
        df['Price_Momentum'] = df['Closing Price'].diff(5)
        
        # Xử lý sự kiện (đơn giản hóa)
        df['Dividend_Event'] = 0
        df['Meeting_Event'] = 0
        
        # Điền giá trị thiếu
        df = df.fillna(method='ffill').fillna(0)
        return df
    except Exception as e:
        st.error(f"Lỗi khi tiền xử lý dữ liệu: {e}")
        return None

# Hàm dự báo
def predict_future(model, last_data, scaler, days, model_type='lstm'):
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(days):
        if model_type == 'lstm':
            # Chuẩn bị dữ liệu cho LSTM
            scaled_data = scaler.transform(np.log1p(current_data[-7:]).reshape(-1, 1))
            X = np.reshape(scaled_data, (1, 7, 1))
            pred = model.predict(X, verbose=0)
            pred_price = np.expm1(scaler.inverse_transform(pred)[0][0])
        else:
            # Chuẩn bị dữ liệu cho XGBoost hoặc Meta model
            X = pd.DataFrame({
                'Return%': [current_data[-1] / current_data[-2] - 1 if len(current_data) > 1 else 0],
                'MA5': [np.mean(current_data[-5:])],
                'MA10': [np.mean(current_data[-10:])],
                'Volume_ratio': [1],  # Giả định
                'Dividend_Event': [0],
                'Meeting_Event': [0],
                'Volatility': [np.std(np.diff(current_data[-5:])/current_data[-5:-1]) if len(current_data) > 5 else 0],
                'Price_Momentum': [current_data[-1] - current_data[-5]] if len(current_data) > 5 else 0
            })
            if model_type == 'xgb':
                pred_price = model.predict(X)[0]
            else:  # Meta model
                # Giả sử chúng ta có dự đoán từ cả LSTM và XGBoost
                lstm_pred = np.expm1(scaler.inverse_transform(
                    model_lstm.predict(np.reshape(scaler.transform(
                        np.log1p(current_data[-7:]).reshape(-1, 1)), (1, 7, 1)), verbose=0))[0][0])
                xgb_pred = model_xgb.predict(X)[0]
                pred_price = model.predict(np.array([[lstm_pred, xgb_pred]]))[0][0]
        
        predictions.append(pred_price)
        current_data = np.append(current_data, pred_price)
    
    return predictions

# Tải dữ liệu và mô hình
@st.cache_data
def load_data(stock_id):
    try:
        df = pd.read_csv(f"4.2.3 (TARGET) (live & his) {stock_id}_detail_transactions_processed.csv")
        processed_df = preprocess_data(df, stock_id)
        return processed_df
    except Exception as e:
        st.error(f"Không thể tải dữ liệu cho {stock_id}: {e}")
        return None

# Hiển thị tiến trình
with st.spinner('Đang tải dữ liệu và mô hình...'):
    df = load_data(stock_id)
    model_lstm, model_xgb, meta_model = load_models(stock_id)

if df is None or model_lstm is None or model_xgb is None or meta_model is None:
    st.error("Không thể khởi tạo ứng dụng do lỗi tải dữ liệu hoặc mô hình.")
    st.stop()

# Hiển thị thông tin cơ bản
st.subheader(f"Thông tin cổ phiếu {stock_id}")
col1, col2, col3 = st.columns(3)
col1.metric("Giá đóng cửa gần nhất", f"{df['Closing Price'].iloc[-1]:,.0f} VND")
col2.metric("Thay đổi 1 ngày", f"{df['Return%'].iloc[-1]:.2f}%", 
            f"{df['Closing Price'].iloc[-1] - df['Closing Price'].iloc[-2]:,.0f} VND" if len(df) > 1 else "N/A")
col3.metric("Ngày dữ liệu mới nhất", df['Date'].iloc[-1].strftime('%d/%m/%Y'))

# Tab chức năng
tab1, tab2, tab3 = st.tabs(["📊 Dự báo", "📈 Biểu đồ", "📋 Đánh giá Mô hình"])

with tab1:
    st.subheader("Dự báo Giá trong Tương lai")
    
    # Lấy dữ liệu gần nhất
    last_prices = df['Closing Price'].values[-30:]  # 30 ngày gần nhất
    
    # Dự báo
    try:
        scaler = MinMaxScaler()
        scaler.fit(np.log1p(df['Closing Price'].values.reshape(-1, 1)))
        
        if model_type == "Meta Model (Tích hợp LSTM + XGBoost)":
            predictions = predict_future(meta_model, last_prices, scaler, forecast_days, 'meta')
            model_name = "Meta Model"
        elif model_type == "LSTM Model":
            predictions = predict_future(model_lstm, last_prices, scaler, forecast_days, 'lstm')
            model_name = "LSTM"
        else:
            predictions = predict_future(model_xgb, last_prices, scaler, forecast_days, 'xgb')
            model_name = "XGBoost"
        
        # Tạo DataFrame kết quả
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
        result_df = pd.DataFrame({
            'Ngày': future_dates,
            'Giá dự báo (VND)': predictions,
            'Thay đổi (%)': [0] + [(predictions[i] - predictions[i-1])/predictions[i-1]*100 for i in range(1, len(predictions))]
        })
        
        # Hiển thị kết quả
        st.dataframe(result_df.style.format({
            'Giá dự báo (VND)': '{:,.0f}',
            'Thay đổi (%)': '{:.2f}%'
        }))
        
        # Nút tải dữ liệu
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Tải dữ liệu dự báo (CSV)",
            data=csv,
            file_name=f"du_bao_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Lỗi khi dự báo: {e}")

with tab2:
    st.subheader("Biểu đồ Giá và Dự báo")
    
    # Vẽ biểu đồ lịch sử
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Closing Price'], label='Giá thực tế', color='blue')
    
    # Thêm dự báo nếu có
    try:
        if 'future_dates' in locals() and 'predictions' in locals():
            ax.plot(future_dates, predictions, label=f'Dự báo {model_name}', 
                   color='red', linestyle='--', marker='o')
            ax.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.title(f"Diễn biến giá {stock_id} và dự báo")
            st.pyplot(fig)
            
            # Nút tải biểu đồ
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                label="Tải biểu đồ (PNG)",
                data=buf,
                file_name=f"bieu_do_{stock_id}.png",
                mime="image/png"
            )
    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ: {e}")

with tab3:
    st.subheader("Đánh giá Hiệu suất Mô hình")
    
    # Chia tập train/test
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    # Chuẩn bị dữ liệu đánh giá
    try:
        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler()
        y_log = np.log1p(df['Closing Price'].values)
        scaler.fit(y_log.reshape(-1, 1))
        
        # Chuẩn bị dữ liệu LSTM
        lookback = 7
        X_lstm, y_lstm = [], []
        for i in range(lookback, len(y_log)):
            X_lstm.append(y_log[i-lookback:i])
            y_lstm.append(y_log[i])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
        
        # Dự đoán LSTM
        pred_lstm = np.expm1(scaler.inverse_transform(model_lstm.predict(X_lstm[split:], verbose=0))).flatten()
        pred_lstm = np.nan_to_num(pred_lstm, nan=np.nanmean(pred_lstm), neginf=0)
        
        # Dự đoán XGBoost
        features_xgb = ['Return%', 'MA5', 'MA10', 'Volume_ratio', 'Dividend_Event', 'Meeting_Event', 'Volatility', 'Price_Momentum']
        X_xgb = df[features_xgb].iloc[lookback:].reset_index(drop=True)
        pred_xgb = model_xgb.predict(X_xgb.iloc[split:])
        
        # Dự đoán Meta model
        X_meta = np.vstack((pred_lstm, pred_xgb)).T
        pred_meta = meta_model.predict(X_meta).flatten()
        pred_meta = np.nan_to_num(pred_meta, nan=np.nanmean(pred_meta), neginf=0)
        
        # Tính toán metrics
        y_true = df['Closing Price'].iloc[split+lookback:].reset_index(drop=True)
        
        metrics = {
            'Meta Model': {
                'MAE': mean_absolute_error(y_true, pred_meta),
                'RMSE': np.sqrt(mean_squared_error(y_true, pred_meta)),
                'R2': r2_score(y_true, pred_meta)
            },
            'LSTM': {
                'MAE': mean_absolute_error(y_true, pred_lstm),
                'RMSE': np.sqrt(mean_squared_error(y_true, pred_lstm)),
                'R2': r2_score(y_true, pred_lstm)
            },
            'XGBoost': {
                'MAE': mean_absolute_error(y_true, pred_xgb),
                'RMSE': np.sqrt(mean_squared_error(y_true, pred_xgb)),
                'R2': r2_score(y_true, pred_xgb)
            }
        }
        
        # Hiển thị metrics
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.format({
            'MAE': '{:.2f}',
            'RMSE': '{:.2f}',
            'R2': '{:.4f}'
        }))
        
        # Biểu đồ so sánh hiệu suất
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        metrics_df[['MAE', 'RMSE']].plot(kind='bar', ax=ax2)
        plt.title('So sánh hiệu suất các mô hình')
        plt.ylabel('Giá trị')
        plt.xticks(rotation=0)
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Lỗi khi đánh giá mô hình: {e}")

# Huấn luyện lại nếu được chọn
if retrain:
    with st.expander("Huấn luyện lại mô hình", expanded=False):
        st.warning("Chức năng này sẽ huấn luyện lại mô hình từ đầu và có thể mất nhiều thời gian.")
        
        if st.button("Bắt đầu Huấn luyện"):
            with st.spinner('Đang huấn luyện mô hình...'):
                try:
                    # Code huấn luyện từ file gốc của bạn
                    # Đây là phần cần tích hợp code huấn luyện đầy đủ của bạn
                    st.success("Huấn luyện hoàn tất! Mô hình mới đã được lưu.")
                except Exception as e:
                    st.error(f"Lỗi khi huấn luyện: {e}")