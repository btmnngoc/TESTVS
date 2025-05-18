import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_growth_chart(df, indicators, company):
    """Tạo biểu đồ đường thể hiện diễn biến các chỉ số theo thời gian"""
    df_filtered = df[df['Indicator'].isin(indicators)]
    
    # Trích xuất đơn vị từ chỉ số đầu tiên
    first_indicator = indicators[0]
    unit = ''
    if '\n' in first_indicator:
        indicator_name, unit = first_indicator.split('\n')
    
    # Tạo figure với layout responsive
    fig = px.line(
        df_filtered,
        x='Period',
        y='Value',
        color='Indicator',
        title=f'Diễn biến các chỉ số tài chính - {company}',
        markers=True,
        height=500  # Tăng chiều cao để có không gian cho legend
    )
    
    # Cập nhật layout tối ưu cho mobile
    fig.update_layout(
        hovermode='x unified',
        yaxis_title=f'Giá trị ({unit})' if unit else 'Giá trị',
        legend_title='Chỉ số',
        legend=dict(
            orientation='h',  # Legend ngang
            yanchor='bottom', # Neo ở dưới
            y=-0.5,           # Đẩy xuống dưới biểu đồ
            xanchor='center', # Căn giữa
            x=0.5,
            font=dict(size=10) # Giảm kích thước font
        ),
        margin=dict(l=20, r=20, t=40, b=120), # Tăng margin dưới cho legend
        xaxis=dict(
            tickangle=-45, # Xoay nhãn trục x để dễ đọc
            tickfont=dict(size=10) # Giảm kích thước font
        ),
        title_x=0.5, # Căn giữa tiêu đề
        title_font_size=16 # Giảm kích thước tiêu đề
    )
    
    # Tùy chỉnh marker và line để dễ phân biệt trên mobile
    fig.update_traces(
        line_width=2,
        marker_size=6
    )
    
    return fig

def create_bar_chart(df, indicators, company, period):
    """Tạo biểu đồ cột so sánh các chỉ số trong một kỳ cụ thể"""
    df_filtered = df[(df['Indicator'].isin(indicators)) & (df['Period'] == period)]
    
    # Sắp xếp theo giá trị
    df_filtered = df_filtered.sort_values('Value', ascending=False)
    
    # Trích xuất đơn vị từ chỉ số đầu tiên
    first_indicator = indicators[0]
    unit = ''
    if '\n' in first_indicator:
        indicator_name, unit = first_indicator.split('\n')
    
    fig = px.bar(
        df_filtered,
        x='Indicator',
        y='Value',
        title=f'Giá trị các chỉ số tài chính - {company} - Kỳ {period}',
        text='Value',
        color='Value',
        color_continuous_scale='Bluered'
    )
    
    fig.update_layout(
        xaxis_title='Chỉ số',
        yaxis_title=f'Giá trị ({unit})' if unit else 'Giá trị',
        coloraxis_showscale=False,
        height=600
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_xaxes(tickangle=45)
    
    return fig

