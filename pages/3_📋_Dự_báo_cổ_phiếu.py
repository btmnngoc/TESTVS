import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from models.stock_prediction import train_stock_prediction_model
import warnings
warnings.filterwarnings('ignore')

# Thiết lập trang
st.set_page_config(page_title="Dự báo giá cổ phiếu ngắn hạn", page_icon="📈", layout="wide")
st.title("💡 Dự báo giá cổ phiếu ngắn hạn")

# Sidebar để chọn cổ phiếu và các tham số
with st.sidebar:
    st.header("Thiết lập dự báo")
    stock_options = ['FPT', 'CMG']
    selected_stock = st.selectbox("Chọn mã cổ phiếu", stock_options)
    
    forecast_days = st.slider("Số ngày dự báo", 1, 30, 7, 
                             help="Chọn số ngày trong tương lai cần dự báo")
    
    model_options = ['Mô hình kết hợp (LSTM + XGBoost)', 'LSTM', 'XGBoost']
    selected_model = st.selectbox("Chọn mô hình", model_options)
    
    st.markdown("---")
    st.markdown("**Lưu ý:**")
    st.info("Dữ liệu được cập nhật tự động từ các nguồn đáng tin cậy. Kết quả dự báo chỉ mang tính chất tham khảo.")

# Hàm tải dữ liệu (có thể tách ra file utils/data_loader.py sau)
@st.cache_data
def load_data(stock_id):
    if stock_id == 'FPT':
        df = pd.read_csv("4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv")
    else:
        df = pd.read_csv("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
    
    # Xử lý dữ liệu như trong code gốc
    df = df[df['StockID'] == stock_id].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')
    df['Closing Price'] = df['Closing Price'].str.replace(',', '').astype(float)
    df['Total Volume'] = df['Total Volume'].str.replace(',', '').astype(float)
    
    return df

# Hàm hiển thị kết quả
def display_results(df, y_test, y_pred, metrics):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Chỉ số đánh giá mô hình")
        st.metric("MAE (Lỗi tuyệt đối trung bình)", f"{metrics['mae']:.2f}")
        st.metric("RMSE (Lỗi bình phương trung bình)", f"{metrics['rmse']:.2f}")
        st.metric("R² (Độ phù hợp)", f"{metrics['r2']:.4f}")
        
        st.markdown("---")
        st.write("**Giải thích chỉ số:**")
        st.info("- MAE: Sai số trung bình giữa giá thực và giá dự báo (càng thấp càng tốt)")
        st.info("- RMSE: Tương tự MAE nhưng phạt nặng hơn các sai số lớn")
        st.info("- R²: Tỷ lệ phương sai được giải thích bởi mô hình (1 là tốt nhất)")
    
    with col2:
        st.subheader("Biểu đồ dự báo")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['Date'].iloc[-len(y_test):], y_test, label='Giá thực tế', linewidth=2)
        ax.plot(df['Date'].iloc[-len(y_test):], y_pred, label='Giá dự báo', linestyle='--', linewidth=2)
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá đóng cửa")
        ax.set_title(f"Dự báo giá {selected_stock} - {selected_model}")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Nút tải biểu đồ
        st.download_button(
            label="Tải biểu đồ",
            data=fig_to_bytes(fig),
            file_name=f"du_bao_{selected_stock}_{datetime.now().strftime('%Y%m%d')}.png",
            mime="image/png"
        )

# Hàm chuyển figure thành bytes để tải xuống
def fig_to_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

# Hàm dự báo tương lai
def forecast_future(model, last_data, scaler, days):
    predictions = []
    current_input = last_data.copy()
    
    for _ in range(days):
        # Dự báo bước tiếp theo
        pred = model.predict(current_input.reshape(1, -1, 1))[0, 0]
        predictions.append(pred)
        
        # Cập nhật input cho bước tiếp theo
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred
    
    # Chuyển đổi lại về giá trị gốc
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    predictions = np.expm1(predictions).flatten()
    
    return predictions

# Main app
def main():
    # Hiển thị thông tin
    st.info(f"Đang tải dữ liệu và huấn luyện mô hình cho cổ phiếu {selected_stock}...")
    
    # Tải dữ liệu
    df = load_data(selected_stock)
    
    # Huấn luyện mô hình (có thể cache lại để tăng tốc độ)
    with st.spinner(f"Đang huấn luyện mô hình cho {selected_stock}..."):
        model, scaler, metrics, y_test, y_pred = train_stock_prediction_model(selected_stock)
    
    # Hiển thị kết quả
    st.success("Hoàn thành huấn luyện mô hình!")
    display_results(df, y_test, y_pred, metrics)
    
    # Dự báo tương lai
    st.subheader(f"Dự báo giá {selected_stock} trong {forecast_days} ngày tới")
    
    # Lấy dữ liệu cuối cùng để dự báo
    last_data = df['Closing Price'].values[-lookback:]
    last_dates = df['Date'].values[-lookback:]
    
    # Chuẩn hóa dữ liệu
    last_data_log = np.log1p(last_data)
    last_data_scaled = scaler.transform(last_data_log.reshape(-1, 1)).flatten()
    
    # Dự báo
    future_predictions = forecast_future(model, last_data_scaled, scaler, forecast_days)
    
    # Tạo dates cho tương lai
    last_date = pd.to_datetime(last_dates[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
    
    # Hiển thị kết quả dự báo
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Chi tiết dự báo:**")
        forecast_df = pd.DataFrame({
            'Ngày': future_dates,
            'Giá dự báo': future_predictions
        })
        st.dataframe(forecast_df.style.format({
            'Giá dự báo': '{:.2f}',
            'Ngày': lambda x: x.strftime('%d/%m/%Y')
        }), hide_index=True)
        
        # Nút tải dữ liệu dự báo
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Tải dữ liệu dự báo (CSV)",
            data=csv,
            file_name=f"du_bao_{selected_stock}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.write("**Biểu đồ dự báo tương lai:**")
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Vẽ dữ liệu lịch sử
        ax.plot(df['Date'].iloc[-30:], df['Closing Price'].iloc[-30:], 
                label='Giá lịch sử', color='blue')
        
        # Vẽ dự báo
        ax.plot(future_dates, future_predictions, 
                label='Dự báo tương lai', color='red', linestyle='--', marker='o')
        
        ax.set_title(f"Dự báo giá {selected_stock} {forecast_days} ngày tới")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá đóng cửa")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
