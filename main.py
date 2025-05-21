import streamlit as st
from PIL import Image

# ============ CẤU HÌNH TRANG ============
st.set_page_config(
    page_title="DABAVERSE - AI Stock Analysis",
    page_icon="🚀",
    layout="wide"
)

# ============ CSS TÙY CHỈNH ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700&family=Space+Grotesk:wght@700&display=swap');
    
    .header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 3rem 0;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #FFFFFF 0%, #E0F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: #94A3B8;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .gradient-bar {
        height: 4px;
        width: 200px;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        border-radius: 2px;
        margin: 1rem auto;
    }
    
    .feature-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #3B82F6;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        color: #3B82F6;
    }
    
    .feature-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 1rem;
    }
    
    .feature-desc {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .welcome-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 3rem;
        margin-bottom: 3rem;
        border: 1px solid #334155;
    }
    
    .welcome-title {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .welcome-text {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 1.1rem;
        line-height: 1.6;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.markdown("""
<div class="header">
    <div class="header-content">
        <h1 class="main-title">DABAVERSE AI TRADING PLATFORM</h1>
        <p class="subtitle">Nền tảng phân tích và dự báo thị trường chứng khoán thế hệ mới</p>
        <div class="gradient-bar"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============ MAIN CONTENT ============
st.markdown("""
<div class="welcome-card">
    <h2 class="welcome-title">Chào mừng đến với DABAVERSE</h2>
    <p class="welcome-text">
        Giải pháp phân tích chứng khoán toàn diện kết hợp trí tuệ nhân tạo và công nghệ phân tích dữ liệu tiên tiến,
        mang đến cái nhìn sâu sắc và dự báo chính xác về thị trường
    </p>
</div>
""", unsafe_allow_html=True)

# ============ FEATURES ============
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <h3 class="feature-title">Dự báo cổ phiếu AI</h3>
        <p class="feature-desc">
            Hệ thống dự báo giá đa mô hình với độ chính xác cao, kết hợp LSTM và XGBoost,
            cung cấp dự báo ngắn hạn với độ tin cậy >85%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h3 class="feature-title">Phân tích kỹ thuật</h3>
        <p class="feature-desc">
            Hơn 50 chỉ báo kỹ thuật từ cơ bản đến nâng cao, nhận diện mô hình giá tự động,
            cảnh báo breakout với độ chính xác cao
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💼</div>
        <h3 class="feature-title">Phân tích cơ bản</h3>
        <p class="feature-desc">
            Đánh giá doanh nghiệp qua 20+ chỉ số tài chính quan trọng, phân tích ngành,
            và dự báo tăng trưởng dài hạn
        </p>
    </div>
    """, unsafe_allow_html=True)
