import streamlit as st

st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Ứng dụng Phân tích và Dự báo Cổ phiếu")

st.markdown("""
    Chào mừng bạn đến với ứng dụng Phân tích và Dự báo Cổ phiếu. Vui lòng chọn chức năng từ menu bên trái.
    
    **Các chức năng chính:**
    - 💡 Dự báo cổ phiếu: Dự báo giá cổ phiếu trong ngắn hạn
    - 📖 Phân tích kỹ thuật: Các chỉ báo kỹ thuật và biểu đồ
    - ✅ Phân tích tài chính: Các chỉ số tài chính cơ bản
""")

st.sidebar.success("Vui lòng chọn chức năng từ menu bên trái.")