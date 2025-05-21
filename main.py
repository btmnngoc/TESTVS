import streamlit as st

# Custom CSS to mimic your Tailwind and inline styles
st.markdown("""
    <style>
        .gradient-text {
            background: linear-gradient(90deg, #FFFFFF 0%, #E0F2FE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .gradient-bar {
            background: linear-gradient(90deg, #3B82F6, #8B5CF6);
            height: 4px;
            width: 192px; /* 48px * 4 (for 4xl container) */
            margin: 0 auto;
            border-radius: 9999px;
        }
        
        .feature-card {
            transition: all 0.3s ease;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid #334155;
            border-radius: 1rem;
            padding: 2rem;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: #3B82F6;
            box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
        }
        
        .welcome-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid #334155;
            border-radius: 1rem;
            padding: 2rem 3rem;
            max-width: 64rem;
            margin: 0 auto 3rem auto;
        }
        
        body {
            background-color: #0F172A;
            color: #F8FAFC;
            font-family: 'Inter', sans-serif;
        }
        
        .header-bg {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            padding: 3rem;
            border-radius: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@700&display=swap');
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <header class="header-bg">
        <div class="text-center">
            <h1 class="main-title gradient-text">DABAVERSE AI TRADING PLATFORM</h1>
            <p class="text-xl text-slate-400 mb-6">Nền tảng phân tích và dự báo thị trường chứng khoán thế hệ mới</p>
            <div class="gradient-bar"></div>
        </div>
    </header>
""", unsafe_allow_html=True)

# Main Content
st.markdown("""
    <main class="container mx-auto px-4 pb-12">
        <!-- Welcome Card -->
        <div class="welcome-card">
            <h2 class="text-3xl font-semibold text-center mb-6">Chào mừng đến với DABAVERSE</h2>
            <p class="text-lg text-slate-400 text-center leading-relaxed">
                Giải pháp phân tích chứng khoán toàn diện kết hợp trí tuệ nhân tạo và công nghệ phân tích dữ liệu tiên tiến,
                mang đến cái nhìn sâu sắc và dự báo chính xác về thị trường
            </p>
        </div>
""", unsafe_allow_html=True)

# Features Section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl text-blue-500 mb-6">
                <i class="fas fa-chart-line"></i>
            </div>
            <h3 class="text-2xl font-semibold mb-4">Dự báo cổ phiếu AI</h3>
            <p class="text-slate-400 leading-relaxed">
                Hệ thống dự báo giá đa mô hình với độ chính xác cao, kết hợp LSTM và XGBoost,
                cung cấp dự báo ngắn hạn với độ tin cậy >85%
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl text-blue-500 mb-6">
                <i class="fas fa-chart-bar"></i>
            </div>
            <h3 class="text-2xl font-semibold mb-4">Phân tích kỹ thuật</h3>
            <p class="text-slate-400 leading-relaxed">
                Hơn 50 chỉ báo kỹ thuật từ cơ bản đến nâng cao, nhận diện mô hình giá tự động,
                cảnh báo breakout với độ chính xác cao
            </p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl text-blue-500 mb-6">
                <i class="fas fa-briefcase"></i>
            </div>
            <h3 class="text-2xl font-semibold mb-4">Phân tích cơ bản</h3>
            <p class="text-slate-400 leading-relaxed">
                Đánh giá doanh nghiệp qua 20+ chỉ số tài chính quan trọng, phân tích ngành,
                và dự báo tăng trưởng dài hạn
            </p>
        </div>
    """, unsafe_allow_html=True)

# Additional Features Section
col4, col5, col6 = st.columns(3)
with col4:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl text-blue-500 mb-6">
                <i class="fas fa-bell"></i>
            </div>
            <h3 class="text-2xl font-semibold mb-4">Cảnh báo thông minh</h3>
            <p class="text-slate-400 leading-relaxed">
                Hệ thống cảnh báo theo thời gian thực khi cổ phiếu đạt các ngưỡng quan trọng,
                giúp bạn không bỏ lỡ cơ hội giao dịch
            </p>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl text-blue-500 mb-6">
                <i class="fas fa-robot"></i>
            </div>
            <h3 class="text-2xl font-semibold mb-4">Tư vấn tự động</h3>
            <p class="text-slate-400 leading-relaxed">
                AI tư vấn giao dịch dựa trên hồ sơ rủi ro cá nhân, đề xuất danh mục đầu tư
                tối ưu theo mục tiêu của bạn
            </p>
        </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl text-blue-500 mb-6">
                <i class="fas fa-globe-asia"></i>
            </div>
            <h3 class="text-2xl font-semibold mb-4">Phân tích thị trường</h3>
            <p class="text-slate-400 leading-relaxed">
                Báo cáo tổng quan thị trường hàng ngày, phân tích xu hướng ngành và tác động
                từ các yếu tố vĩ mô
            </p>
        </div>
    """, unsafe_allow_html=True)

# Call to Action
st.markdown("""
    <div class="mt-16 text-center">
        <h2 class="text-3xl font-bold mb-6">Bắt đầu hành trình đầu tư thông minh ngay hôm nay</h2>
        <a href="#" class="bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold py-3 px-8 rounded-full hover:shadow-lg transition-all duration-300 transform hover:scale-105 inline-block">
            Đăng ký ngay
        </a>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <footer class="bg-slate-900 py-8 mt-16">
        <div class="container mx-auto px-4">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <div class="mb-4 md:mb-0">
                    <h3 class="text-xl font-bold">DABAVERSE</h3>
                    <p class="text-slate-400">Nền tảng AI phân tích chứng khoán hàng đầu</p>
                </div>
                <div class="flex space-x-6">
                    <a href="#" class="text-slate-400 hover:text-white transition-colors">
                        <i class="fab fa-facebook-f text-xl"></i>
                    </a>
                    <a href="#" class="text-slate-400 hover:text-white transition-colors">
                        <i class="fab fa-twitter text-xl"></i>
                    </a>
                    <a href="#" class="text-slate-400 hover:text-white transition-colors">
                        <i class="fab fa-linkedin-in text-xl"></i>
                    </a>
                    <a href="#" class="text-slate-400 hover:text-white transition-colors">
                        <i class="fab fa-youtube text-xl"></i>
                    </a>
                </div>
            </div>
            <div class="border-t border-slate-800 mt-8 pt-8 text-center text-slate-500">
                <p>© 2023 DABAVERSE. All rights reserved.</p>
            </div>
        </div>
    </footer>
""", unsafe_allow_html=True)
