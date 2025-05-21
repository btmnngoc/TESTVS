import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from utils.data_loader import load_data, clean_data
from utils.visualization import create_growth_chart, create_bar_chart
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

# Tiêu đề trang
st.markdown("""
    <header class="header-bg">
        <div class="text-center">
            <div class="flex justify-center mb-6">
                <div class="relative">
                    <i class="fas fa-chart-line text-6xl gradient-text glow floating"></i>
                    <i class="fas fa-star text-xl text-yellow-400 absolute -top-2 -right-2"></i>
                </div>
            </div>
            <h1 class="main-title text-5xl font-bold mb-4 gradient-text">DABAFIN AI TRADING</h1>
            <p class="text-xl text-slate-300 mb-6">Nền tảng phân tích và dự báo thị trường chứng khoán thế hệ mới</p>
            <div class="gradient-bar"></div>
        </div>
    </header>
""", unsafe_allow_html=True)

# Tải dữ liệu
df = load_data("6.2 (his) financialreport_metrics_FPT_CMG_processed.csv")
df_long = clean_data(df)

# Lọc dữ liệu theo công ty
company = st.selectbox("Chọn công ty", df_long['StockID'].unique(), index=0)
df_company = df_long[df_long['StockID'] == company]

# Nhóm chỉ số
indicator_groups = {
    'Tất cả các chỉ số': [],
    'Khả năng sinh lời': [
        'Tỷ suất sinh lợi trên tổng tài sản bình quân (ROAA)',
        'Tỷ suất lợi nhuận trên vốn chủ sở hữu bình quân (ROEA)',
        'Tỷ suất lợi nhuận gộp biên',
        'Tỷ suất sinh lợi trên doanh thu thuần'
    ],
    'Khả năng thanh toán': [
        'Tỷ số thanh toán hiện hành (ngắn hạn)',
        'Tỷ số thanh toán nhanh',
        'Tỷ số thanh toán bằng tiền mặt'
    ],
    'Đòn bẩy tài chính': [
        'Tỷ số Nợ trên Tổng tài sản',
        'Tỷ số Nợ trên Vốn chủ sở hữu'
    ],
    'Hiệu quả hoạt động': [
        'Vòng quay tổng tài sản (Hiệu suất sử dụng toàn bộ tài sản)',
        'Vòng quay hàng tồn kho',
        'Vòng quay phải thu khách hàng'
    ],
    'Chỉ số thị trường': [
        'Chỉ số giá thị trường trên thu nhập (P/E)',
        'Chỉ số giá thị trường trên giá trị sổ sách (P/B)',
        'Beta'
    ]
}

# Chọn nhóm chỉ số
selected_group = st.selectbox("Chọn nhóm chỉ số", list(indicator_groups.keys()))

# Lọc chỉ số theo nhóm đã chọn
if selected_group == 'Tất cả các chỉ số':
    available_indicators = df_company['Indicator'].unique()
else:
    available_indicators = [ind for ind in df_company['Indicator'].unique() 
                           if any(g in ind for g in indicator_groups[selected_group])]

# Chọn chỉ số cụ thể
selected_indicators = st.multiselect(
    "Chọn chỉ số muốn phân tích",
    available_indicators,
    default=available_indicators[:min(4, len(available_indicators))]
)

# Tạo tab cho các loại biểu đồ
tab1, tab2, tab3 = st.tabs(["Biểu đồ đường", "Biểu đồ cột", "Tăng trưởng kết hợp"])

with tab1:
    if selected_indicators:
        fig_line = create_growth_chart(df_company, selected_indicators, company)
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một chỉ số để hiển thị biểu đồ.")

with tab2:
    if selected_indicators:
        period = st.selectbox("Chọn kỳ báo cáo", df_company['Period'].unique())
        fig_bar = create_bar_chart(df_company, selected_indicators, company, period)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một chỉ số để hiển thị biểu đồ.")
with tab3:
    if len(selected_indicators) >= 2:
        st.subheader("🔍 So sánh Tốc độ Tăng trưởng")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            base_period = st.selectbox(
                "Chọn kỳ làm mốc", 
                df_company['Period'].unique(),
                index=len(df_company['Period'].unique())-4,
                key='base_period_select'
            )
        with col2:
            show_values = st.checkbox("Hiển thị giá trị", value=True)
        
        # Tính toán tăng trưởng
        @st.cache_data
        def calculate_growth(df, indicators, base_period):
            growth_data = []
            for indicator in indicators:
                df_ind = df[df['Indicator'] == indicator]
                base_value = df_ind[df_ind['Period'] == base_period]['Value'].values[0]
                
                for _, row in df_ind.iterrows():
                    growth = (row['Value'] - base_value) / base_value * 100 if base_value != 0 else 0
                    growth_data.append({
                        'Indicator': row['Indicator'].split('\n')[0],  # Bỏ đơn vị trong tên chỉ số
                        'Period': row['Period'],
                        'Growth (%)': growth,
                        'Value': row['Value']  # Giữ lại giá trị gốc để hiển thị
                    })
            return pd.DataFrame(growth_data)
        
        df_growth = calculate_growth(df_company, selected_indicators, base_period)
        
        # Tạo biểu đồ
        fig_growth = px.line(
            df_growth,
            x='Period',
            y='Growth (%)',
            color='Indicator',
            title=f'📈 Tốc độ tăng trưởng so với kỳ {base_period}',
            markers=True,
            hover_data=['Value'],
            height=500
        )
        
        # Hiển thị giá trị trên biểu đồ nếu được chọn
        if show_values:
            fig_growth.update_traces(
                text=df_growth['Growth (%)'].round(2).astype(str) + '%',
                textposition='top center',
                textfont=dict(size=10)
            )
        
        # Cải tiến layout
        fig_growth.update_layout(
            hovermode='x unified',
            yaxis_title='Tăng trưởng (%)',
            legend_title='Chỉ số',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.5,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(l=50, r=50, t=80, b=120),
            xaxis=dict(tickangle=-45),
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Arial'
            )
        )
        
        # Thêm đường zero để dễ so sánh
        fig_growth.add_hline(
            y=0, 
            line_dash='dot', 
            line_color='gray',
            annotation_text=f"Mốc {base_period}", 
            annotation_position='bottom right'
        )
        
        # Hiển thị biểu đồ
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Hiển thị bảng dữ liệu
        with st.expander("📊 Xem dữ liệu chi tiết"):
    # Tạo danh sách period theo đúng thứ tự thời gian
            period_order = [
        'Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023',
        'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024'
    ]
    
    # Lọc chỉ những period có trong dữ liệu
        available_periods = [p for p in period_order if p in df_growth['Period'].unique()]
    
    # Tạo pivot table và sắp xếp cột theo thứ tự thời gian
        pivot_df = df_growth.pivot_table(
            index='Indicator',
            columns='Period',
            values='Growth (%)',
            aggfunc='first'
        )[available_periods]  # Chỉ lấy các cột có trong dữ liệu và sắp xếp đúng thứ tự
    
    # Định dạng và hiển thị bảng
        st.dataframe(
            pivot_df.style.format('{:.2f}%')
                    .background_gradient(cmap='RdYlGn', axis=1)
                    .set_properties(**{
                        'text-align': 'center',
                        'font-size': '12px'
                    }),
            use_container_width=True,
        )        
     # Tự động điều chỉnh chiều cao
    else:
        st.warning("⚠️ Cần chọn ít nhất 2 chỉ số để so sánh tốc độ tăng trưởng.")
        st.info("ℹ️ Vui lòng chọn thêm chỉ số từ danh sách bên trên")
