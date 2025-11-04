import streamlit as st
import requests
from PIL import Image
import io
import base64
import datetime

# ====== THÔNG TIN MODEL ROBOFLOW ======
ROBOFLOW_API_KEY = "nWA6ayjI5bGNpXkkbsAb"  # <-- THAY Ở ĐÂY
ROBOFLOW_MODEL_URL = "https://serverless.roboflow.com"  # <-- THAY Ở ĐÂY

st.set_page_config(page_title="BKAI - Crack Detection", layout="wide")

st.title("BKAI - AI phát hiện vết nứt bê tông")
st.write("Upload ảnh bê tông, hệ thống sẽ phân tích bằng mô hình AI (CNN/Mask R-CNN).")

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

    # Chuyển ảnh sang bytes để gửi lên Roboflow
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    with st.spinner("Đang gửi ảnh tới mô hình AI, vui lòng đợi..."):
        # Gửi request tới Roboflow
        response = requests.post(
            ROBOFLOW_MODEL_URL,
            params={"api_key": ROBOFLOW_API_KEY},
            data=img_bytes,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

    if response.status_code == 200:
        result = response.json()
        st.subheader("Kết quả phân tích")

        predictions = result.get("predictions", [])

        if len(predictions) == 0:
            st.success("✅ Không phát hiện vết nứt (No crack detected).")
            has_crack = False
        else:
            has_crack = True
            st.error(f"⚠ Phát hiện {len(predictions)} vùng có khả năng là vết nứt.")
            # Hiển thị chi tiết từng vùng nứt
            for i, pred in enumerate(predictions, start=1):
                x = pred.get("x")
                y = pred.get("y")
                w = pred.get("width")
                h = pred.get("height")
                conf = pred.get("confidence")
                cls = pred.get("class", "crack")

                st.write(
                    f"- Vết nứt {i}: lớp `{cls}`, độ tin cậy: {conf:.2f}, "
                    f"tọa độ tâm ({x:.0f}, {y:.0f}), kích thước ({w:.0f}x{h:.0f})"
                )

        # Phân tích nguyên nhân (gợi ý)
        st.subheader("Phân tích nguyên nhân (gợi ý)")
        if has_crack:
            st.write(
                "Dựa trên hình dạng và mật độ vết nứt, nguyên nhân có thể do co ngót bê tông, "
                "ảnh hưởng tải trọng hoặc tác động môi trường. Cần kết hợp thêm thông tin hiện trường "
                "để đánh giá chính xác hơn."
            )
        else:
            st.write("Chưa ghi nhận vùng nứt rõ ràng, tuy nhiên vẫn cần kiểm tra thực tế hiện trường.")

        # ====== LƯU LOG (ĐƠN GIẢN) ======
        log_line = f"{datetime.datetime.now()};{name};{email};{has_crack};{note}\n"
        with open("logs.csv", "a", encoding="utf-8") as f:
            f.write(log_line)

        st.info("Thông tin phân tích đã được lưu lại (demo trong file logs.csv trên server).")
    else:
        st.error("Có lỗi khi gọi API mô hình AI. Vui lòng kiểm tra lại API key / URL.")
elif submitted and uploaded_file is None:
    st.warning("Vui lòng chọn một ảnh trước khi bấm Phân tích.")

