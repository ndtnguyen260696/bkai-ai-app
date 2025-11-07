import os
import streamlit as st
import requests
from PIL import Image, ImageDraw

# =========================================================
# 1. CẤU HÌNH ROBOFLOW + LOGO BKAI
# =========================================================

# Thay đường dẫn Roboflow bằng của bạn
ROBOFLOW_FULL_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=THAY_API_KEY_VÀO_ĐÂY"

# Logo BKAI - từ URL chính thức
BKAI_LOGO = "https://bkai.b12sites.com/index"

def show_bkai_logo():
    """
    Hiển thị logo BKAI từ URL hoặc file local một cách an toàn.
    Không để app bị crash nếu lỗi đọc ảnh.
    """
    try:
        # Nếu là URL → kiểm tra khả năng tải
        if BKAI_LOGO.startswith("http"):
            response = requests.get(BKAI_LOGO, timeout=5)
            if response.status_code == 200 and "text/html" not in response.headers.get("Content-Type", ""):
                # Nếu link trả ảnh trực tiếp (image/png hoặc image/jpeg)
                st.image(BKAI_LOGO, caption="BKAI", use_column_width=True)
            else:
                # Nếu URL không phải ảnh (ví dụ trang HTML), hiển thị fallback
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:10px;">
                        <a href="{BKAI_LOGO}" target="_blank" style="text-decoration:none;">
                            <h3 style="color:#a78bfa;">🌐 BKAI Website</h3>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        # Nếu là file local → kiểm tra tồn tại
        elif os.path.exists(BKAI_LOGO):
            st.image(BKAI_LOGO, caption="BKAI", use_column_width=True)
        else:
            st.info("Không tìm thấy file logo BKAI (bỏ qua).")
    except Exception as e:
        st.warning(f"Không thể hiển thị logo BKAI ({e}).")

# =========================================================
# 2. GIAO DIỆN VÀ GỌI LOGO
# =========================================================

# Giao diện sidebar
with st.sidebar:
    show_bkai_logo()
    st.markdown("### ⚙️ Cấu hình phân tích")
    min_conf = st.slider(
        "Ngưỡng confid
