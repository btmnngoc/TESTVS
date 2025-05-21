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
    
    # Các tính năng chính
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <h3 class="feature-title">Dự báo cổ phiếu</h3>
            <p class="feature-desc">
                Dự báo giá cổ phiếu trong ngắn hạn với độ chính xác cao sử dụng mô hình AI đa tầng
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📖</div>
            <h3 class="feature-title">Phân tích kỹ thuật</h3>
            <p class="feature-desc">
                Hơn 50 chỉ báo kỹ thuật và công cụ vẽ biểu đồ chuyên nghiệp
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">✅</div>
            <h3 class="feature-title">Phân tích tài chính</h3>
            <p class="feature-desc">
                Đánh giá doanh nghiệp qua các chỉ số tài chính cơ bản và nâng cao
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo sản phẩm
    st.markdown("""
    <div class="demo-section">
        <h3 class="demo-title">TRẢI NGHIỆM SỨC MẠNH DABAVERSE</h3>
        <div class="demo-grid">
            <div class="demo-item">
                <img src="https://i.imgur.com/abc123.png" class="demo-img">
                <p>Giao diện dự báo AI</p>
            </div>
            <div class="demo-item">
                <img src="https://i.imgur.com/def456.png" class="demo-img">
                <p>Biểu đồ phân tích</p>
            </div>
            <div class="demo-item">
                <img src="https://i.imgur.com/ghi789.png" class="demo-img">
                <p>Báo cáo tài chính</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============ MAIN APP ============
def main():
    render_header()
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()
