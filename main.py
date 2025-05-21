import streamlit as st

# Inject CSS styles into Streamlit
st.markdown("""
    <style>
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        @keyframes gradient {
            0% { background-position: 0% center; }
            50% { background-position: 100% center; }
            100% { background-position: 0% center; }
        }
        
        .gradient-text {
            background: linear-gradient(90deg, #8A2BE2 0%, #00BFFF 50%, #7CFC00 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-size: 200% auto;
            animation: gradient 3s ease infinite;
        }
        
        .gradient-bar {
            background: linear-gradient(90deg, #8A2BE2, #00BFFF, #7CFC00);
            background-size: 200% auto;
            animation: gradient 3s ease infinite;
            height: 4px;
            width: 192px;
            margin: 0 auto;
            border-radius: 9999px;
        }
        
        .feature-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(138, 43, 226, 0.3);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 2rem;
            transition: all 0.3s ease;
            margin: 0.5rem; /* Thêm margin để cách nhau */
        }
        
        .feature-card:hover {
            transform: translateY(-5px) scale(1.02);
            border-color: #00BFFF;
            box-shadow: 0 10px 30px rgba(0, 191, 255, 0.3);
        }
        
        .welcome-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(0, 191, 255, 0.3);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 2rem 3rem;
            max-width: 64rem;
            margin: 0 auto 3rem auto;
        }
        
        body {
            background-color: #020617;
            color: #F8FAFC;
            font-family: 'Inter', sans-serif;
        }
        
        .header-bg {
            background: linear-gradient(135deg, #020617 0%, #0F172A 100%);
            box-shadow: 0 4px 30px rgba(0, 191, 255, 0.1);
            padding: 4rem;
            border-radius: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            text-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
        }
        
        .floating {
            animation: float 3s ease-in-out infinite;
        }
        
        .glow {
            filter: drop-shadow(0 0 8px currentColor);
        }
        
        .gradient-button {
            background: linear-gradient(90deg, #8A2BE2 0%, #00BFFF 50%, #7CFC00 100%);
            background-size: 200% auto;
            transition: all 0.3s ease;
            border-radius: 9999px;
            padding: 0.75rem 2rem;
            color: white;
            font-weight: 600;
            display: inline-block;
        }
        
        .gradient-button:hover {
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 10px 20px rgba(0, 191, 255, 0.4);
            background-position: right center;
        }
        
        .footer-bg {
            background: linear-gradient(180deg, rgba(2, 6, 23, 0) 0%, #020617 100%);
            padding: 3rem 0;
            margin-top: 6rem;
        }
        
        .card-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 2rem; /* Tăng khoảng cách giữa các thẻ */
            padding: 1rem;
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@700&display=swap');
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <header class="header-bg">
        <div class="text-center">
            <div class="flex justify-center mb-6">
                <div class="relative">
                    <i class="fas fa-chart-line text-6xl gradient-text glow floating"></i>
                    <i class="fas fa-star text-xl text-yellow-400 absolute -top-2 -right-2"></i>
                </div>
            </div>
            <h1 class="main-title text-5xl font-bold mb-4 gradient-text">DABAVERSE AI TRADING</h1>
            <p class="text-xl text-slate-300 mb-6">Nền tảng phân tích và dự báo thị trường chứng khoán thế hệ mới</p>
            <div class="gradient-bar"></div>
        </div>
    </header>
""", unsafe_allow_html=True)

# Main Content
st.markdown("""
    <main class="container mx-auto px-4 pb-12">
        <!-- Welcome Card -->
        <div class="welcome-card">
            <h2 class="text-3xl font-semibold text-center mb-6 gradient-text">Chào mừng đến với DABAVERSE</h2>
            <p class="text-lg text-slate-300 text-center leading-relaxed">
                Giải pháp phân tích chứng khoán toàn diện kết hợp trí tuệ nhân tạo và công nghệ phân tích dữ liệu tiên tiến,
                mang đến cái nhìn sâu sắc và dự báo chính xác về thị trường
            </p>
            <div class="flex justify-center mt-8">
                <div class="relative">
                    <div class="absolute -inset-1 bg-gradient-to-r from-purple-600 to-blue-500 rounded-lg blur opacity-75"></div>
                    <div class="relative px-6 py-2 bg-slate-900 rounded-lg">
                        <span class="text-slate-300">Độ chính xác dự báo: <span class="font-bold text-green-400">87.3%</span></span>
                    </div>
                </div>
            </div>
        </div>
    </main>
""", unsafe_allow_html=True)

# Features Section with improved layout
st.markdown('<div class="card-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl mb-6 glow"><i class="fas fa-chart-line gradient-text"></i></div>
            <h3 class="text-2xl font-semibold mb-4 gradient-text">Dự báo cổ phiếu AI</h3>
            <p class="text-slate-300 leading-relaxed">
                Hệ thống dự báo giá đa mô hình với độ chính xác cao, kết hợp LSTM và XGBoost,
                cung cấp dự báo ngắn hạn với độ tin cậy >85%
            </p>
            <div class="mt-4">
                <span class="inline-block bg-blue-900/50 text-blue-300 text-xs px-2 py-1 rounded">Deep Learning</span>
                <span class="inline-block bg-purple-900/50 text-purple-300 text-xs px-2 py-1 rounded">Time Series</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl mb-6 glow"><i class="fas fa-chart-bar gradient-text"></i></div>
            <h3 class="text-2xl font-semibold mb-4 gradient-text">Phân tích kỹ thuật</h3>
            <p class="text-slate-300 leading-relaxed">
                Hơn 50 chỉ báo kỹ thuật từ cơ bản đến nâng cao, nhận diện mô hình giá tự động,
                cảnh báo breakout với độ chính xác cao
            </p>
            <div class="mt-4">
                <span class="inline-block bg-green-900/50 text-green-300 text-xs px-2 py-1 rounded">RSI</span>
                <span class="inline-block bg-yellow-900/50 text-yellow-300 text-xs px-2 py-1 rounded">MACD</span>
                <span class="inline-block bg-red-900/50 text-red-300 text-xs px-2 py-1 rounded">Bollinger</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl mb-6 glow"><i class="fas fa-briefcase gradient-text"></i></div>
            <h3 class="text-2xl font-semibold mb-4 gradient-text">Phân tích cơ bản</h3>
            <p class="text-slate-300 leading-relaxed">
                Đánh giá doanh nghiệp qua 20+ chỉ số tài chính quan trọng, phân tích ngành,
                và dự báo tăng trưởng dài hạn
            </p>
            <div class="mt-4">
                <span class="inline-block bg-indigo-900/50 text-indigo-300 text-xs px-2 py-1 rounded">P/E</span>
                <span class="inline-block bg-pink-900/50 text-pink-300 text-xs px-2 py-1 rounded">ROE</span>
                <span class="inline-block bg-teal-900/50 text-teal-300 text-xs px-2 py-1 rounded">EPS</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Additional Features Section with improved layout
st.markdown('<div class="card-container">', unsafe_allow_html=True)
col4, col5, col6 = st.columns([1, 1, 1], gap="medium")
with col4:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl mb-6 glow"><i class="fas fa-bell gradient-text"></i></div>
            <h3 class="text-2xl font-semibold mb-4 gradient-text">Cảnh báo thông minh</h3>
            <p class="text-slate-300 leading-relaxed">
                Hệ thống cảnh báo theo thời gian thực khi cổ phiếu đạt các ngưỡng quan trọng,
                giúp bạn không bỏ lỡ cơ hội giao dịch
            </p>
            <div class="mt-4">
                <span class="inline-block bg-orange-900/50 text-orange-300 text-xs px-2 py-1 rounded">Real-time</span>
                <span class="inline-block bg-blue-900/50 text-blue-300 text-xs px-2 py-1 rounded">Alerts</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl mb-6 glow"><i class="fas fa-robot gradient-text"></i></div>
            <h3 class="text-2xl font-semibold mb-4 gradient-text">Tư vấn tự động</h3>
            <p class="text-slate-300 leading-relaxed">
                AI tư vấn giao dịch dựa trên hồ sơ rủi ro cá nhân, đề xuất danh mục đầu tư
                tối ưu theo mục tiêu của bạn
            </p>
            <div class="mt-4">
                <span class="inline-block bg-purple-900/50 text-purple-300 text-xs px-2 py-1 rounded">AI Advisor</span>
                <span class="inline-block bg-green-900/50 text-green-300 text-xs px-2 py-1 rounded">Portfolio</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
        <div class="feature-card">
            <div class="text-4xl mb-6 glow"><i class="fas fa-globe-asia gradient-text"></i></div>
            <h3 class="text-2xl font-semibold mb-4 gradient-text">Phân tích thị trường</h3>
            <p class="text-slate-300 leading-relaxed">
                Báo cáo tổng quan thị trường hàng ngày, phân tích xu hướng ngành và tác động
                từ các yếu tố vĩ mô
            </p>
            <div class="mt-4">
                <span class="inline-block bg-red-900/50 text-red-300 text-xs px-2 py-1 rounded">Macro</span>
                <span class="inline-block bg-yellow-900/50 text-yellow-300 text-xs px-2 py-1 rounded">Trends</span>
                <span class="inline-block bg-teal-900/50 text-teal-300 text-xs px-2 py-1 rounded">Reports</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Market Data Visualization
st.markdown("""
    <div class="mt-16 p-8 rounded-2xl bg-gradient-to-br from-slate-900/50 to-blue-900/30 border border-blue-900/30">
        <h2 class="text-3xl font-bold mb-6 gradient-text text-center">Thống kê thị trường</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-slate-900/50 p-4 rounded-lg border border-blue-900/30 text-center">
                <div class="text-2xl text-blue-400 mb-2"><i class="fas fa-arrow-up"></i> 1,256</div>
                <div class="text-sm text-slate-400">Cổ phiếu tăng</div>
            </div>
            <div class="bg-slate-900/50 p-4 rounded-lg border border-red-900/30 text-center">
                <div class="text-2xl text-red-400 mb-2"><i class="fas fa-arrow-down"></i> 874</div>
                <div class="text-sm text-slate-400">Cổ phiếu giảm</div>
            </div>
            <div class="bg-slate-900/50 p-4 rounded-lg border border-green-900/30 text-center">
                <div class="text-2xl text-green-400 mb-2"><i class="fas fa-chart-line"></i> 1.8%</div>
                <div class="text-sm text-slate-400">Tăng trưởng TB</div>
            </div>
            <div class="bg-slate-900/50 p-4 rounded-lg border border-purple-900/30 text-center">
                <div class="text-2xl text-purple-400 mb-2"><i class="fas fa-bolt"></i> 92%</div>
                <div class="text-sm text-slate-400">Độ chính xác AI</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Call to Action
st.markdown("""
    <div class="mt-16 text-center">
        <h2 class="text-3xl font-bold mb-6 gradient-text">Bắt đầu hành trình đầu tư thông minh ngay hôm nay</h2>
        <p class="text-lg text-slate-300 mb-8 max-w-2xl mx-auto">
            Đăng ký ngay để trải nghiệm sức mạnh của AI trong phân tích thị trường chứng khoán
        </p>
        <a href="#" class="gradient-button">Đăng ký ngay</a>
        <div class="mt-8 flex justify-center space-x-4">
            <div class="flex items-center">
                <i class="fas fa-check-circle text-green-400 mr-2"></i>
                <span class="text-sm text-slate-300">Dùng thử miễn phí 7 ngày</span>
            </div>
            <div class="flex items-center">
                <i class="fas fa-shield-alt text-blue-400 mr-2"></i>
                <span class="text-sm text-slate-300">Bảo mật dữ liệu</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <footer class="footer-bg">
        <div class="container mx-auto px-4">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
                <div>
                    <h3 class="text-xl font-bold mb-4 gradient-text">DABAVERSE</h3>
                    <p class="text-slate-400 mb-4">Nền tảng AI phân tích chứng khoán hàng đầu</p>
                    <div class="flex space-x-4">
                        <a href="#" class="text-slate-400 hover:text-white"><i class="fab fa-facebook-f text-lg"></i></a>
                        <a href="#" class="text-slate-400 hover:text-white"><i class="fab fa-twitter text-lg"></i></a>
                        <a href="#" class="text-slate-400 hover:text-white"><i class="fab fa-linkedin-in text-lg"></i></a>
                        <a href="#" class="text-slate-400 hover:text-white"><i class="fab fa-youtube text-lg"></i></a>
                    </div>
                </div>
                <div>
                    <h4 class="text-lg font-semibold mb-4 text-white">Sản phẩm</h4>
                    <ul class="space-y-2">
                        <li><a href="#" class="text-slate-400 hover:text-blue-400">Dự báo AI</a></li>
                        <li><a href="#" class="text-slate-400 hover:text-blue-400">Phân tích kỹ thuật</a></li>
                        <li><a href="#" class="text-slate-400 hover:text-blue-400">Phân tích cơ bản</a></li>
                        <li><a href="#" class="text-slate-400 hover:text-blue-400">Tư vấn tự động</a></li>
                    </ul>
                </div>
                <div>
                    <h4 class="text-lg font-semibold mb-4 text-white">Tài nguyên</h4>
                    <ul class="space-y-2">
                        <li><a href="#" class="text-slate-400 hover:text-blue-400">Tài liệu hướng dẫn</a></li>
                        <li><a href="#" class="text-slate-400 hover:text-blue-400">Blog phân tích</a></li>
                        <li><a href="#" class="text-slate-400 hover:text-blue-400">Webinar</a></li>
                        <li><a href="#" class="text-slate-400 hover:text-blue-400">API</a></li>
                    </ul>
                </div>
                <div>
                    <h4 class="text-lg font-semibold mb-4 text-white">Liên hệ</h4>
                    <ul class="space-y-2">
                        <li class="text-slate-400"><i class="fas fa-envelope mr-2 text-blue-400"></i> support@dabaverse.com</li>
                        <li class="text-slate-400"><i class="fas fa-phone mr-2 text-blue-400"></i> +84 123 456 789</li>
                        <li class="text-slate-400"><i class="fas fa-map-marker-alt mr-2 text-blue-400"></i> Hà Nội, Việt Nam</li>
                    </ul>
                </div>
            </div>
            <div class="border-t border-slate-800 pt-8 text-center text-slate-500">
                <p>© 2023 DABAVERSE. All rights reserved.</p>
            </div>
        </div>
    </footer>
""", unsafe_allow_html=True)
