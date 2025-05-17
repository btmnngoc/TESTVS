import streamlit as st

st.set_page_config(
    page_title="Phân tích Chỉ số Tài chính",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Phân tích Chỉ số Tài chính Doanh nghiệp")
st.markdown("""
Phân tích các chỉ số tài chính quan trọng theo thời gian, đánh giá tăng trưởng và so sánh giữa các doanh nghiệp.
""")

st.sidebar.success("Chọn chức năng phân tích từ menu.")
