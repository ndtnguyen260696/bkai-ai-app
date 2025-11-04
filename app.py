import streamlit as st
from PIL import Image
import io
import datetime

from inference_sdk import InferenceHTTPClient

# ===============================
# CẤU HÌNH ROBOFLOW – LẤY Y HỆT
# TRONG HÌNH BẠN GỬI
# ===============================

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="nWA6ayjI5bGNpXkkbsAb"  # API key của bạn
)

MODEL_ID = "crack_segmentation_detection/4"  # Model ID + version


# ===============================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ===============================

st.set_page_config(page_title="BKAI - Crack Segmentation", layout="wide")

st.title("BKAI – Công nghệ AI phát hiện và phân đoạn vết nứt bê tông")
st.write("Upload ảnh bê tông, mô hình CNN sẽ phân tích và trả về kết quả.")

# ===== FORM NHẬP THÔNG TIN NGƯỜI DÙNG =====
with st.form("upload_form"):
    name = st.text_input("Họ tên (Name)")
    email = st.text_input("Email")
    note = st.text_area("Ghi chú về ảnh / công trình (Note)")
    uploaded_file = st.file_uploader(
        "Chọn ảnh bê tông (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    submitted = st.form_submit_button("Phân tích ảnh")

# ===============================
# XỬ LÝ ẢNH KHI BẤM “PHÂN TÍCH”
# ===============================
if submitted and uploaded_file is not None:
    # Hiển thị ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    # Chuyển ảnh sang bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    image_bytes = buf.read()

    st.write("Đang gửi ảnh tới mô hình AI trên Roboflow...")

    try:
        # Gửi ảnh tới Roboflow bằng inference-sdk
        result = CLIENT.infer(
            image=image_bytes,
            model_id=MODEL_ID
        )
    except Exception as e:
        st.error(f"Lỗi khi gọi API mô hình: {e}")
    else:
        st.subheader("Kết quả từ mô hình")
        st.json(result)

        predictions = result.get("predictions", [])
        st.write(f"Số vùng nứt được phát hiện: **{len(predictions)}**")

        # Lưu/hiển thị log đơn giản
        st.write("---")
        st.write("Thông tin người dùng:")
        st.write(f"- Tên: {name}")
        st.write(f"- Email: {email}")
        st.write(f"- Ghi chú: {note}")
        st.write(f"- Thời gian phân tích: {datetime.datetime.now()}")

elif submitted and uploaded_file is None:
    st.warning("Vui lòng chọn một ảnh bê tông trước khi bấm 'Phân tích ảnh'.")



