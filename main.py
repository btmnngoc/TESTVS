import streamlit as st
from PIL import Image
import base64

# ============ CẤU HÌNH TRANG ============
st.set_page_config(
    page_title="DABAVERSE - AI Stock Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ CSS TÙY CHỈNH ============
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # Tạo file CSS riêng

# ============ HEADER ============
def render_header():
    st.markdown("""
    <div class="header">
        <div class="header-content">
            <h1 class="main-title">DABAVERSE <span class="ai-text">AI</span> TRADING PLATFORM</h1>
            <p class="subtitle">Công cụ phân tích và dự báo thị trường chứng khoán thế hệ mới</p>
            <div class="gradient-bar"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============ MAIN CONTENT ============
def render_main_content():
    st.markdown("""
    <div class="welcome-card">
        <h2 class="welcome-title">Chào mừng đến với nền tảng DABAVERSE</h2>
        <p class="welcome-text">
            Giải pháp phân tích chứng khoán toàn diện kết hợp trí tuệ nhân tạo và phân tích dữ liệu nâng cao
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical Analysis Section
    st.markdown("""
    <div class="feature-card">
        <h3 class="feature-title">Phân tích kỹ thuật cổ phiếu</h3>
        <p class="feature-desc">
            Tổng quan biểu đồ giá, Bollinger Bands, MACD & RSI, và dự liệu
        </p>
        <div class="chart-section">
            <label>Chọn cổ phiếu:</label>
            <select>
                <option value="FPT">FPT</option>
            </select>
            <label>Số ngày hiện thị:</label>
            <input type="range" min="30" max="365" value="90">
            <h4>Tổng quan FPT</h4>
            <p>Giá hiện tại: 135,900.00</p>
            <p>Thay đổi 1 ngày: -1,600.00 (-1.19%)</p>
            <p>Tổng biên độ: 42,900.00 (+39.03%)</p>
            <img src="https://example.com/chart.png" alt="Technical Chart">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Financial Overview Section
    st.markdown("""
    <div class="feature-card">
        <h3 class="feature-title">Hệ thống Dự báo Giá cổ phiếu FPT & CMG</h3>
        <p class="feature-desc">
            Dự báo giá cổ phiếu dựa trên mô hình hybrid LSTM, XGBoost, và Meta-model
        </p>
        <div class="forecast-section">
            <h4>Dự báo Giá FPT</h4>
            <p>Biểu đồ Dự báo</p>
            <img src="https://example.com/forecast.png" alt="Forecast Chart">
            <table>
                <tr><th>Ngày</th><th>Giá dự báo</th><th>Biến động (%)</th></tr>
                <tr><td>2025-03-13 00:00:00</td><td>136,598</td><td>+0.51%</td></tr>
                <tr><td>2025-03-14 00:00:00</td><td>136,732</td><td>+0.61%</td></tr>
                <tr><td>2025-03-15 00:00:00</td><td>136,856</td><td>+0.70%</td></tr>
            </table>
            <p>Biến động tổng tại điểm: 1.00%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock Performance Section
    st.markdown("""
    <div class="feature-card">
        <h3 class="feature-title">So sánh Tốc độ Tăng trưởng</h3>
        <p class="feature-desc">
            Hiển thị giai đoạn Q1 2023
        </p>
        <div class="performance-section">
            <img src="https://example.com/performance.png" alt="Performance Chart">
            <table>
                <tr><th>Chỉ số</th><th>Q1 2023</th><th>Q2 2023</th><th>Q3 2023</th><th>Q4 2023</th><th>Q1 2024</th><th>Q2 2024</th><th>Q3 2024</th></tr>
                <tr><td>Beta</td><td>0.00%</td><td>1.96%</td><td>-7.84%</td><td>-0.89%</td><td>-8.82%</td><td>15.69%</td><td>16.67%</td></tr>
                <tr><td>Tỷ suất lợi nhuận trên vốn chủ sở hữu (ROCE)</td><td>0.00%</td><td>-16.41%</td><td>45.90%</td><td>-32.52%</td><td>-7.29%</td><td>-24.92%</td><td>34.04%</td></tr>
                <tr><td>Tỷ số Ngắn hạn trên Tổng nợ phải trả</td><td>0.00%</td><td>-2.88%</td><td>-1.87%</td><td>-3.09%</td><td>-3.78%</td><td>-3.75%</td><td>-2.82%</td></tr>
                <tr><td>Tỷ số Ngắn hạn trên Tổng nợ phải trả</td><td>0.00%</td><td>-7.32%</td><td>0.13%</td><td>-13.41%</td><td>-9.92%</td><td>-0.96%</td><td>-6.58%</td></tr>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============ MAIN APP ============
def main():
    render_header()
    render_main_content()

if __name__ == "__main__":
    main()
