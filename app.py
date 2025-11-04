import streamlit as st
import requests
from PIL import Image
import io
import datetime

# ==========================
# PASTE URL ROBOFLOW Ở ĐÂY
# ==========================
# Ví dụ: "https://detect.roboflow.com/crack_segmentation_detection/1?api_key=XXXX"
ROBOFLOW_FULL_URL = "https://serverless.roboflow.com/ten_project/1?api_key=nWA6ayjI5bGNpXkkbsAb"
st.set_page_config(page_title="BKAI - Crack Segmentation", layout="wide")

st.title("BKAI - AI phát hiện và phân đoạn vết nứt bê tông")
st.write("Upload ảnh bê tông, mô hình AI (instance segmentation) sẽ phân tích và trả về kết quả.")

# ====== FORM NHẬP THÔNG TIN NGƯỜI DÙNG ======
with st.form("upload_form"):
    name = st.text_input("Họ tên (Name)")
    email = st.text_input("Email")
    note = st.text_area("Ghi chú về ảnh / công trình (Note)")
    uploaded_file = st.file_uploader("Chọn ảnh bê tông (JPG/PNG)", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Phân tích ảnh")

if submitted and uploaded_file is not None:
    # Hiển thị ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    # Chuyển ảnh sang bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    with st.spinner("Đang gửi ảnh tới mô hình AI, vui lòng đợi..."):
        # Gửi request tới Roboflow
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        try:
            response = requests.post(ROBOFLOW_FULL_URL, files=files)
        except Exception as e:
            st.error(f"Lỗi kết nối tới Roboflow: {e}")
            st.stop()

    # ====== DEBUG: Hiển thị mã lỗi nếu có ======
    if response.status_code != 200:
        st.error("Có lỗi khi gọi API mô hình AI. Vui lòng kiểm tra lại URL trong ROBOFLOW_FULL_URL.")
        st.code(f"Status code: {response.status_code}\nResponse text: {response.text}")
    else:
        result = response.json()
        st.subheader("Kết quả phân tích (raw JSON)")
        st.json(result)

        predictions = result.get("predictions", [])

        if len(predictions) == 0:
            st.success("✅ Không phát hiện vết nứt (No crack detected).")
            has_crack = False
        else:
            has_crack = True
            st.error(f"⚠ Phát hiện {len(predictions)} vùng vết nứt.")
            # In thông tin cơ bản
            for i, pred in enumerate(predictions, start=1):
                conf = pred.get("confidence", 0)
                cls = pred.get("class", "crack")
                st.write(f"- Mask {i}: lớp `{cls}`, độ tin cậy: {conf:.2f}")

        st.subheader("Phân tích nguyên nhân (gợi ý)")
        if has_crack:
            st.write(
                "Dựa trên hình dạng và mật độ vết nứt, nguyên nhân có thể do co ngót bê tông, "
                "ảnh hưởng tải trọng hoặc tác động môi trường. Cần kết hợp thêm thông tin hiện trường "
                "để đánh giá chính xác hơn."
            )
        else:
            st.write("Chưa ghi nhận vùng nứt rõ ràng, tuy nhiên vẫn cần kiểm tra thực tế hiện trường.")

        # Lưu log đơn giản trên server
        log_line = f"{datetime.datetime.now()};{name};{email};{has_crack};{note}\n"
        with open("logs.csv", "a", encoding="utf-8") as f:
            f.write(log_line)

elif submitted and uploaded_file is None:
    st.warning("Vui lòng chọn một ảnh trước khi bấm Phân tích.")







