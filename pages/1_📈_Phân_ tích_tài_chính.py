import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from utils.data_loader import load_data, clean_data
from utils.visualization import create_growth_chart, create_bar_chart

# Tiêu đề trang
st.markdown(
    """
    <div style='text-align: center; border-bottom: 1px solid #ccc; padding-bottom: 10px;'>
        <div style='font-size: 2.8rem; font-weight: 900; color: #FD6200;'>DABFIN</div> 
        <div style='font-size: 2.5rem; font-weight: 900; color: #0E6994;'>PHÂN TÍCH TÀI CHÍNH DOANH NGHIỆP</div>
    </div>
    """,
    unsafe_allow_html=True)
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