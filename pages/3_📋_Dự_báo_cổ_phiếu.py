import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from models.stock_prediction import train_stock_prediction_model
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dự báo giá cổ phiếu ngắn hạn",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .metric-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("💡 Dự báo giá cổ phiếu ngắn hạn")

# Sidebar configuration
with st.sidebar:
    st.header("Thiết lập dự báo")
    
    # Stock selection
    stock_options = ['FPT', 'CMG']
    selected_stock = st.selectbox(
        "Chọn mã cổ phiếu",
        stock_options,
        help="Chọn mã cổ phiếu bạn muốn dự báo"
    )
    
    # Forecast period
    forecast_days = st.slider(
        "Số ngày dự báo",
        1, 30, 7,
        help="Chọn số ngày trong tương lai cần dự báo"
    )
    
    # Model selection
    model_options = ['Mô hình kết hợp (LSTM + XGBoost)', 'LSTM', 'XGBoost']
    selected_model = st.selectbox(
        "Chọn mô hình",
        model_options,
        index=0,
        help="Chọn mô hình dự báo"
    )
    
    # Retrain option
    force_retrain = st.checkbox(
        "Huấn luyện lại mô hình",
        value=False,
        help="Bỏ chọn để sử dụng mô hình đã lưu (nếu có)"
    )
    
    st.markdown("---")
    st.markdown("**Lưu ý:**")
    st.info("""
        - Dữ liệu được cập nhật tự động từ các nguồn đáng tin cậy
        - Kết quả dự báo chỉ mang tính chất tham khảo
        - Mô hình sẽ tự động lưu sau khi huấn luyện
    """)

# Helper functions
def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for download"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def forecast_future(models, scaler, last_data, lookback, days, model_type='combined'):
    """Generate future forecasts"""
    # Prepare input data
    last_data_log = np.log1p(last_data)
    last_data_scaled = scaler.transform(last_data_log.reshape(-1, 1)).flatten()
    current_input = last_data_scaled[-lookback:].copy()
    predictions = []
    
    for _ in range(days):
        # Prepare input based on model type
        if model_type == 'LSTM':
            lstm_input = current_input.reshape(1, lookback, 1)
            next_pred_scaled = models["lstm"].predict(lstm_input, verbose=0)[0, 0]
        elif model_type == 'XGBoost':
            # For XGBoost, we need to create features for the future prediction
            # This is simplified - in practice you'd need to update all features
            next_pred_scaled = models["xgb"].predict(current_input.reshape(1, -1))[0]
        else:  # Combined model
            lstm_input = current_input.reshape(1, lookback, 1)
            lstm_pred = models["lstm"].predict(lstm_input, verbose=0)[0, 0]
            
            # For XGBoost part, we'd need the actual features - this is simplified
            xgb_pred = models["xgb"].predict(current_input.reshape(1, -1))[0]
            
            # Combine predictions
            meta_input = np.array([[lstm_pred, xgb_pred]])
            next_pred_scaled = models["meta"].predict(meta_input, verbose=0)[0, 0]
        
        predictions.append(next_pred_scaled)
        current_input = np.roll(current_input, -1)
        current_input[-1] = next_pred_scaled
    
    # Convert predictions to original scale
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    predictions = np.expm1(predictions).flatten()
    
    return predictions

# Main app function
def main():
    # Show loading message
    with st.spinner(f"Đang tải mô hình cho {selected_stock}..."):
        models, scaler, lookback, metrics, y_test, y_pred, df = train_stock_prediction_model(
            selected_stock, force_retrain=force_retrain
        )
    
    # Show success message
    st.success(f"""
        Hoàn thành tải mô hình! 
        Dữ liệu huấn luyện đến ngày: {metrics['last_training_date']}
    """)
    
    # Display model metrics
    st.subheader("📊 Kết quả đánh giá mô hình")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE (Lỗi tuyệt đối trung bình)", f"{metrics['mae']:.2f}")
    with col2:
        st.metric("RMSE (Lỗi bình phương trung bình)", f"{metrics['rmse']:.2f}")
    with col3:
        st.metric("R² (Độ phù hợp)", f"{metrics['r2']:.4f}")
    
    # Display evaluation chart
    st.subheader("📈 Biểu đồ đánh giá mô hình")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'].iloc[-len(y_test):], y_test, label='Giá thực tế', linewidth=2)
    ax.plot(df['Date'].iloc[-len(y_test):], y_pred, label='Giá dự báo', linestyle='--', linewidth=2)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá đóng cửa")
    ax.set_title(f"Kết quả dự báo cho {selected_stock}")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Download button for evaluation chart
    st.download_button(
        label="Tải biểu đồ đánh giá",
        data=fig_to_bytes(fig),
        file_name=f"evaluation_{selected_stock}_{datetime.now().strftime('%Y%m%d')}.png",
        mime="image/png"
    )
    
    # Future forecast section
    st.subheader("🔮 Dự báo giá tương lai")
    
    # Get last available data for forecasting
    last_prices = df['Closing Price'].values[-lookback:]
    last_date = df['Date'].iloc[-1]
    
    # Generate forecast
    if selected_model == 'LSTM':
        model_type = 'LSTM'
    elif selected_model == 'XGBoost':
        model_type = 'XGBoost'
    else:
        model_type = 'combined'
    
    future_prices = forecast_future(
        models, scaler, last_prices, lookback, forecast_days, model_type
    )
    
    # Create future dates
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
    
    # Display forecast results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Chi tiết dự báo:**")
        forecast_df = pd.DataFrame({
            'Ngày': future_dates,
            'Giá dự báo': future_prices,
            'Thay đổi %': np.concatenate([
                [np.nan],
                (future_prices[1:] - future_prices[:-1]) / future_prices[:-1] * 100
            ])
        })
        
        # Format the display
        st.dataframe(
            forecast_df.style.format({
                'Ngày': lambda x: x.strftime('%d/%m/%Y'),
                'Giá dự báo': '{:.2f}',
                'Thay đổi %': '{:.2f}%'
            }).applymap(
                lambda x: 'color: green' if isinstance(x, str) and '+' in x else (
                    'color: red' if isinstance(x, str) and '-' in x else ''
                ),
                subset=['Thay đổi %']
            ),
            hide_index=True,
            use_container_width=True
        )
        
        # Download forecast data
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Tải dữ liệu dự báo (CSV)",
            data=csv,
            file_name=f"forecast_{selected_stock}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("**Biểu đồ dự báo:**")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Plot historical data (last 30 days)
        ax2.plot(
            df['Date'].iloc[-30:], 
            df['Closing Price'].iloc[-30:], 
            label='Giá lịch sử', 
            color='blue',
            linewidth=2
        )
        
        # Plot forecast
        ax2.plot(
            future_dates, 
            future_prices, 
            label='Dự báo tương lai', 
            color='red', 
            linestyle='--',
            marker='o',
            linewidth=2
        )
        
        ax2.set_title(f"Dự báo giá {selected_stock} {forecast_days} ngày tới")
        ax2.set_xlabel("Ngày")
        ax2.set_ylabel("Giá đóng cửa")
        ax2.legend()
        ax2.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        
        # Download forecast chart
        st.download_button(
            label="Tải biểu đồ dự báo",
            data=fig_to_bytes(fig2),
            file_name=f"forecast_chart_{selected_stock}_{datetime.now().strftime('%Y%m%d')}.png",
            mime="image/png"
        )
    
    # Model explanation section
    st.markdown("---")
    st.subheader("ℹ️ Giải thích mô hình")
    
    st.markdown("""
    **Mô hình kết hợp (LSTM + XGBoost):**
    - Sử dụng sức mạnh của cả hai mô hình LSTM (cho dữ liệu chuỗi thời gian) và XGBoost (cho đặc trưng tĩnh)
    - Mô hình meta học cách kết hợp tốt nhất các dự báo từ hai mô hình cơ sở
    - Thường cho kết quả chính xác nhất trong các thử nghiệm
    
    **Mô hình LSTM:**
    - Mạng neural đặc biệt cho dữ liệu chuỗi thời gian
    - Hiệu quả trong việc học các mẫu phức tạp trong dữ liệu giá cổ phiếu
    
    **Mô hình XGBoost:**
    - Mô hình dựa trên cây quyết định được tối ưu hóa
    - Hiệu quả trong việc học từ các đặc trưng tài chính và kỹ thuật
    """)

if __name__ == "__main__":
    main()