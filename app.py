import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import datetime
import math

# =========================================================
# 1. CẤU HÌNH URL ROBOFLOW
#    → BẮT BUỘC phải sửa dòng dưới cho đúng dự án của bạn
#    Vào Roboflow: Project → Deploy → Hosted API → Python
#    Copy nguyên URL dạng:
#    https://detect.roboflow.com/<model_id>/<version>?api_key=<API_KEY>
# =========================================================
ROBOFLOW_FULL_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"


# =========================================================
# 2. HÀM VẼ KHUNG VÀ ĐƯỜNG NỨT
# =========================================================
def draw_predictions(image: Image.Image, predictions, min_conf: float = 0.0) -> Image.Image:
    """
    Vẽ:
      - Khung đỏ quanh vết nứt (bounding box)
      - Đường nứt (polyline) nếu JSON có trường 'points'
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for i, p in enumerate(predictions):
        conf = float(p.get("confidence", 0))
        if conf < min_conf:
            continue

        x = p.get("x")
        y = p.get("y")
        w = p.get("width")
        h = p.get("height")

        if None in (x, y, w, h):
            continue

        # Roboflow dùng x,y là tâm box
        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        # Khung đỏ
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

        # Ghi nhãn: crack (0.91)
        cls = p.get("class", "crack")
        label = f"{cls} ({conf:.2f})"
        # Vẽ nền label đơn giản
        text_x, text_y = x0 + 3, y0 + 3
        draw.text((text_x, text_y), label, fill="red")

        # Thử vẽ đường nứt nếu có 'points'
        points = p.get("points")
        if points:
            flat_points = []

            # points kiểu dict: {"0-100":[[x,y],...], "100-200":[...],...}
            if isinstance(points, dict):
                # Duyệt theo thứ tự key để đường được liền
                for k in sorted(points.keys()):
                    segment = points[k]
                    if isinstance(segment, list):
                        for pt in segment:
                            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                                flat_points.append(tuple(pt))

            # points kiểu list trực tiếp: [[x,y],[x,y],...]
            elif isinstance(points, list):
                for pt in points:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        flat_points.append(tuple(pt))

            # Vẽ đường vàng theo polyline
            if len(flat_points) >= 2:
                draw.line(flat_points, fill="yellow", width=2)

    return overlay


# =========================================================
# 3. GIAO DIỆN STREAMLIT
# =========================================================
st.set_page_config(page_title="BKAI - Crack Segmentation", layout="wide")

st.title("BKAI – Công nghệ AI phát hiện và phân tích vết nứt bê tông")
st.write(
    """
Ứng dụng này cho phép bạn **upload ảnh bê tông**, mô hình AI sẽ:
- Phát hiện các vùng có vết nứt
- Vẽ khung và đường crack lên ảnh
- Hiển thị biểu đồ **độ tin cậy (confidence)** của từng vết nứt
"""
)

# Thanh bên: cấu hình
st.sidebar.header("Cấu hình phân tích")
min_conf = st.sidebar.slider(
    "Ngưỡng độ tin cậy tối thiểu (confidence)", 0.0, 1.0, 0.3, 0.05
)
st.sidebar.write("Chỉ hiển thị các vết nứt có confidence ≥", round(min_conf, 2))

# Form nhập
with st.form("upload_form"):
    name = st.text_input("Họ và tên (tùy chọn)")
    email = st.text_input("Email (tùy chọn)")
    note = st.text_area("Ghi chú về ảnh / công trình (tùy chọn)")
    uploaded_file = st.file_uploader("Chọn ảnh bê tông (JPG/PNG)", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Phân tích ảnh")

# =========================================================
# 4. XỬ LÝ SAU KHI NGƯỜI DÙNG NHẤN "PHÂN TÍCH ẢNH"
# =========================================================
if submitted:
    if uploaded_file is None:
        st.warning("Vui lòng chọn một ảnh trước khi bấm **Phân tích ảnh**.")
    else:
        # Đọc ảnh
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Không đọc được ảnh: {e}")
            st.stop()

        # Hiển thị ảnh gốc & chuẩn bị layout 2 cột
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_column_width=True)

        st.info("Đang gửi ảnh tới Roboflow, vui lòng chờ vài giây...")

        # Chuyển ảnh sang bytes
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        # Gửi request tới Roboflow
        try:
            response = requests.post(
                ROBOFLOW_FULL_URL,
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi khi gọi API Roboflow: {e}")
            st.stop()

        # Kiểm tra mã trả về
        if response.status_code != 200:
            st.error("Roboflow trả về lỗi. Kiểm tra lại URL & API key trong ROBOFLOW_FULL_URL.")
            st.write(f"Status code: {response.status_code}")
            st.text(response.text[:1500])
            st.stop()

        # Parse JSON
        try:
            result = response.json()
        except Exception as e:
            st.error(f"Không parse được JSON trả về: {e}")
            st.text(response.text[:2000])
            st.stop()

        # Hiển thị JSON raw để debug / nghiên cứu
        with st.expander("Xem chi tiết JSON (kết quả raw từ model)", expanded=False):
            st.json(result)

        predictions = result.get("predictions", [])

        # Lọc theo confidence
        filtered_preds = [p for p in predictions if float(p.get("confidence", 0)) >= min_conf]

        # =====================================================
        # 4.1. ẢNH CÓ ĐÁNH DẤU VẾT NỨT
        # =====================================================
        annotated = draw_predictions(image, filtered_preds, min_conf=min_conf)

        with col2:
            st.subheader("Ảnh có đánh dấu vết nứt")
            if len(filtered_preds) == 0:
                st.image(image, use_column_width=True)
                st.info("Không có vết nứt nào đạt ngưỡng confidence đã chọn.")
            else:
                st.image(annotated, use_column_width=True)

        # =====================================================
        # 4.2. THỐNG KÊ & BIỂU ĐỒ ĐỘ TIN CẬY
        # =====================================================
        st.subheader("Thống kê kết quả phân tích")

        total_found = len(predictions)
        total_used = len(filtered_preds)

        if total_found == 0:
            st.write("🔍 **Model không phát hiện vết nứt nào trong ảnh này.**")
        else:
            confidences = [float(p.get("confidence", 0)) for p in predictions]
            max_conf = max(confidences)
            min_conf_pred = min(confidences)
            avg_conf = sum(confidences) / len(confidences)

            st.markdown(
                f"""
- Tổng số vết nứt model phát hiện: **{total_found}**
- Số vết nứt hiển thị (confidence ≥ {min_conf:.2f}): **{total_used}**
- Độ tin cậy cao nhất: **{max_conf:.2f}**
- Độ tin cậy thấp nhất: **{min_conf_pred:.2f}**
- Độ tin cậy trung bình: **{avg_conf:.2f}**
                """
            )

            # Chuẩn bị dữ liệu vẽ biểu đồ cột
            chart_data = {
                "Crack ID": [f"Crack {i+1}" for i in range(len(confidences))],
                "Confidence": confidences,
            }

            st.write("### Biểu đồ độ tin cậy của từng vết nứt")
            st.bar_chart(
                data={"Confidence": confidences},
                x=None,
                y="Confidence"
            )
            st.caption("Mỗi cột tương ứng với một vết nứt, trục Y là confidence (0–1).")

        # =====================================================
        # 4.3. HIỂN THỊ THÔNG TIN NGƯỜI DÙNG (LOG)
        # =====================================================
        st.write("---")
        st.subheader("Thông tin phiên phân tích")
        st.write(f"- Thời gian: **{datetime.datetime.now()}**")
        if name:
            st.write(f"- Người dùng: **{name}**")
        if email:
            st.write(f"- Email: **{email}**")
        if note:
            st.write(f"- Ghi chú: {note}")








