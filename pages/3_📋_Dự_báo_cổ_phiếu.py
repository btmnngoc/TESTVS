import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from tensorflow.keras import losses


# Thiết lập trang
st.set_page_config(page_title="Dự báo giá cổ phiếu", page_icon="📈", layout="wide")
st.title("📈 Dự báo giá cổ phiếu FPT và CMG")

# Hàm tiền xử lý dữ liệu
def preprocess_data(stock_id):
    # Load dữ liệu giao dịch
    if stock_id == 'FPT':
        df = pd.read_csv("4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv")
    else:
        df = pd.read_csv("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
    
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
    df = df.fillna(0)

    # Tích hợp dữ liệu sự kiện
    df_dividend = pd.read_csv("3.2 (live & his) news_dividend_issue (FPT_CMG)_processed.csv")
    df_meeting = pd.read_csv("3.3 (live & his) news_shareholder_meeting (FPT_CMG)_processed.csv")

    df_dividend_stock = df_dividend[df_dividend['StockID'] == stock_id].copy()
    df_meeting_stock = df_meeting[df_meeting['StockID'] == stock_id].copy()
    df_dividend_stock.loc[:, 'Execution Date'] = pd.to_datetime(df_dividend_stock['Execution Date'], format='%d/%m/%Y', errors='coerce')
    df_meeting_stock.loc[:, 'Execution Date'] = pd.to_datetime(df_meeting_stock['Execution Date'], format='%d/%m/%Y')

    df['Dividend_Event'] = df['Date'].isin(df_dividend_stock['Execution Date']).astype(int)
    df['Meeting_Event'] = df['Date'].isin(df_meeting_stock['Execution Date']).astype(int)

    # Tích hợp dữ liệu tài chính
    df_financial = pd.read_csv("6.5 (his) financialreport_metrics_Nhóm ngành_Công nghệ thông tin (of FPT_CMG)_processed.csv")
    
    def clean_financial_data(df):
        df['Indicator'] = df['Indicator'].str.replace('\n', '', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip()
        for col in df.columns[3:]:
            df[col] = df[col].str.replace(',', '').astype(float, errors='ignore')
        return df
    
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
    
    df_financial_stock = df_financial[(df_financial['Stocks'].str.contains(stock_id)) & (df_financial['Indicator'].isin(indicators))].copy()
    
    quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023', 'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
    df_financial_melted = df_financial_stock.melt(id_vars=['Indicator'], value_vars=quarters, var_name='Quarter', value_name='Value')
    
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

# Hàm dự báo
def predict_stock_price(stock_id, days_to_predict=7):
    # Load dữ liệu
    df = preprocess_data(stock_id)
    
    # Load các model đã train
    custom_objects = {
        'MeanSquaredError': MeanSquaredError,
        'Adam': Adam
    }
    
    if stock_id == 'FPT':
        model_lstm = load_model('models/lstm_model_FPT.h5', 
                              custom_objects=custom_objects)
        meta_model = load_model('models/meta_model_FPT.h5',
                              custom_objects=custom_objects)
    else:
        model_lstm = load_model('models/lstm_model_CMG.h5',
                              custom_objects=custom_objects)
        meta_model = load_model('models/meta_model_CMG.h5',
                              custom_objects=custom_objects)
    
    # Chuẩn bị dữ liệu cho dự báo
    features_xgb = ['Return%', 'MA5', 'MA10', 'Volume_ratio', 'Dividend_Event', 'Meeting_Event', 'Volatility', 'Price_Momentum'] + [
        'Tỷ suất lợi nhuận trên Vốn chủ sở hữu bình quân (ROEA)%',
        'Tỷ lệ lãi EBIT%',
        'Chỉ số giá thị trường trên giá trị sổ sách (P/B)Lần',
        'Chỉ số giá thị trường trên thu nhập (P/E)Lần',
        'P/SLần',
        'Tỷ suất sinh lợi trên vốn dài hạn bình quân (ROCE)%',
        'Thu nhập trên mỗi cổ phần (EPS)VNĐ'
    ]
    
    X_xgb = df[features_xgb]
    y = df['Closing Price']
    
    # Chuẩn hóa dữ liệu cho LSTM
    y_log = np.log1p(df['Closing Price'])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(y_log.values.reshape(-1, 1))
    
    lookback = 7
    X_lstm = []
    for i in range(lookback, len(scaled_data)):
        X_lstm.append(scaled_data[i - lookback:i, 0])
    X_lstm = np.array(X_lstm)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    
    # Dự báo với các model
    pred_lstm = np.expm1(scaler.inverse_transform(model_lstm.predict(X_lstm[-1].reshape(1, lookback, 1)))).flatten()[0]
    pred_xgb = model_xgb.predict(X_xgb.iloc[-1:].values.reshape(1, -1))[0]
    
    # Dự báo tổng hợp
    meta_input = np.array([[pred_lstm, pred_xgb]])
    final_pred = meta_model.predict(meta_input).flatten()[0]
    
    # Tạo dữ liệu dự báo cho nhiều ngày
    last_date = df['Date'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
    forecast_prices = [final_pred * (1 + 0.002*i) for i in range(days_to_predict)]  # Giả định tăng nhẹ
    
    return df, forecast_dates, forecast_prices

# Giao diện Streamlit
st.sidebar.header("Tùy chọn dự báo")
stock_option = st.sidebar.selectbox("Chọn mã cổ phiếu", ['FPT', 'CMG'])
days_to_predict = st.sidebar.slider("Số ngày dự báo", 1, 30, 7)

if st.sidebar.button("Dự báo giá"):
    with st.spinner(f'Đang dự báo giá {stock_option}...'):
        try:
            historical_data, forecast_dates, forecast_prices = predict_stock_price(stock_option, days_to_predict)
            
            # Hiển thị kết quả
            st.subheader(f"Kết quả dự báo cho {stock_option}")
            
            # Tạo DataFrame cho dữ liệu dự báo
            forecast_df = pd.DataFrame({
                'Ngày': forecast_dates,
                'Giá dự báo (VND)': [round(price, 2) for price in forecast_prices]
            })
            
            # Hiển thị bảng dự báo
            st.dataframe(forecast_df.style.format({
                'Giá dự báo (VND)': '{:,.2f}'
            }), use_container_width=True)
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Dữ liệu lịch sử
            ax.plot(historical_data['Date'][-30:], historical_data['Closing Price'][-30:], 
                    label='Giá lịch sử', color='blue', marker='o')
            
            # Dữ liệu dự báo
            ax.plot(forecast_dates, forecast_prices, 
                    label='Giá dự báo', color='red', linestyle='--', marker='x')
            
            ax.set_title(f'Diễn biến giá và dự báo {stock_option}')
            ax.set_xlabel('Ngày')
            ax.set_ylabel('Giá (VND)')
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Hiển thị thông tin thống kê
            st.subheader("Thống kê giá gần đây")
            col1, col2, col3 = st.columns(3)
            
            last_price = historical_data['Closing Price'].iloc[-1]
            min_30d = historical_data['Closing Price'][-30:].min()
            max_30d = historical_data['Closing Price'][-30:].max()
            
            col1.metric("Giá đóng cửa gần nhất", f"{last_price:,.2f} VND")
            col2.metric("Giá thấp nhất 30 ngày", f"{min_30d:,.2f} VND")
            col3.metric("Giá cao nhất 30 ngày", f"{max_30d:,.2f} VND")
            
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {str(e)}")

# Hiển thị thông tin về mô hình
st.sidebar.markdown("---")
st.sidebar.subheader("Thông tin mô hình")
st.sidebar.markdown("""
- **Mô hình kết hợp**: LSTM + XGBoost
- **Dữ liệu sử dụng**:
  - Giá lịch sử
  - Khối lượng giao dịch
  - Sự kiện cổ tức
  - Sự kiện đại hội cổ đông
  - Chỉ số tài chính
""")

# Hướng dẫn sử dụng
st.expander("Hướng dẫn sử dụng").markdown("""
1. Chọn mã cổ phiếu cần dự báo (FPT hoặc CMG)
2. Chọn số ngày muốn dự báo (từ 1 đến 30 ngày)
3. Nhấn nút "Dự báo giá" để xem kết quả
4. Kết quả bao gồm:
   - Bảng giá dự báo chi tiết
   - Biểu đồ so sánh giá lịch sử và giá dự báo
   - Các thống kê giá gần đây
""")