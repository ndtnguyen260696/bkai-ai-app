import streamlit as st
import requests
from PIL import Image
import io
import datetime

# ===============================
# CẤU HÌNH ROBOFLOW
# ===============================

# Model ID và version trên Roboflow (theo hình bạn gửi: crack_segmentation_detection/4)
ROBOFLOW_MODEL_ID = "crack_segmentation_detection/4"

# API key của bạn trên Roboflow
ROBOFLOW_API_KEY = "nWAGayjl5bGNpXkkbsAb"

# URL gọi API đúng chuẩn (KHÔNG dùng link universe.roboflow.com)
ROBOFLOW_FULL_URL = (
    f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"
    f"?api_key={ROBOFLOW_API_KEY}"
)

# Cấu hình trang Streamlit
st.set_page_config(page_title="BKAI - Crack Segmentation", layout="wide")

st.title("BKAI – Công nghệ AI phát hiện và phân đoạn vết nứt bê tông")
st.write("Upload ảnh bê tông, mô hình CNN sẽ phân tích và trả về kết quả.")

# ===============================
# FORM NHẬP THÔNG TIN NGƯỜI DÙNG
# ===============================
with st.form("upload_form"):
    name = st.text_input("Họ tên (Name)")
    email = st.text_input("Email")
    note = st.text_area("Ghi chú về ảnh / công trình (Note)")
    uploaded_file = st.file_uploader(
        "Chọn ảnh bê tông (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    submitted = st.form_submit_button("Phân tích ảnh")

# ===============================
# XỬ LÝ KHI NGƯỜI DÙNG BẤM "PHÂN TÍCH ẢNH"
# ===============================
if submitted and uploaded_file is not None:
    # Hiển thị ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    # Chuyển ảnh sang bytes (JPEG)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    st.write("Đang gửi ảnh tới mô hình AI trên Roboflow...")

    # Gửi request tới Roboflow
    try:
        response = requests.post(
            ROBOFLOW_FULL_URL,
            files={"file": ("image.jpg", img_bytes, "image/jpeg")},
            timeout=30,
        )
    except Exception as e:
        st.error(f"Lỗi khi gọi API mô hình: {e}")
    else:
        if response.status_code != 200:
            st.error("Có lỗi khi gọi API mô hình AI. Vui lòng kiểm tra lại cấu hình.")
            st.write(f"Status code: {response.status_code}")
            # Hiển thị một phần nội dung trả về để debug
            st.text(response.text[:1000])
        else:
            result = response.json()

            st.subheader("Kết quả từ mô hình")
            st.json(result)

            # Tóm tắt nhanh số lượng vết nứt phát hiện được (nếu có trường predictions)
            predictions = result.get("predictions", [])
            st.write(f"Số vùng nứt được phát hiện: **{len(predictions)}**")

            # Lưu log đơn giản
            st.write("---")
            st.write("Thông tin người dùng:")
            st.write(f"- Tên: {name}")
            st.write(f"- Email: {email}")
            st.write(f"- Ghi chú: {note}")
            st.write(f"- Thời gian phân tích: {datetime.datetime.now()}")
elif submitted and uploaded_file is None:
    st.warning("Vui lòng chọn một ảnh bê tông trước khi bấm 'Phân tích ảnh'.")


