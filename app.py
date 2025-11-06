import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import datetime

# =========================================================
# 1. CẤU HÌNH ROBOFLOW
#    → BẮT BUỘC: sửa dòng dưới cho đúng model của bạn
#    Vào Roboflow: Project → Deploy → Hosted API → Python
#    Copy nguyên URL dạng:
#    https://detect.roboflow.com/<model_id>/<version>?api_key=<API_KEY>
# =========================================================
ROBOFLOW_FULL_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"


# =========================================================
# 2. HÀM VẼ KHUNG VÀ POLYLINE VẾT NỨT
# =========================================================
def extract_poly_points(points_field):
    """
    Chuyển trường 'points' trong JSON thành list [(x,y), ...]
    Hỗ trợ:
      - dict: {"0-100":[[x,y],...], "100-200":[...], ...}
      - list trực tiếp: [[x,y],[x,y],...]
    """
    flat = []

    if isinstance(points_field, dict):
        for k in sorted(points_field.keys()):
            seg = points_field[k]
            if isinstance(seg, list):
                for pt in seg:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        flat.append((pt[0], pt[1]))

    elif isinstance(points_field, list):
        for pt in points_field:
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                flat.append((pt[0], pt[1]))

    return flat


def draw_predictions(image: Image.Image, predictions, min_conf: float = 0.0) -> Image.Image:
    """
    Vẽ:
      - Khung tím quanh vùng nứt (bounding box)
      - Đường polyline tím theo 'points' nếu có
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for p in predictions:
        conf = float(p.get("confidence", 0))
        if conf < min_conf:
            continue

        x = p.get("x")
        y = p.get("y")
        w = p.get("width")
        h = p.get("height")
        if None in (x, y, w, h):
            continue

        # Roboflow: x,y là tâm box
        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        # Khung tím
        draw.rectangle([x0, y0, x1, y1], outline="#A020F0", width=3)

        cls = p.get("class", "crack")
        label = f"{cls} ({conf:.2f})"
        draw.text((x0 + 3, y0 + 3), label, fill="#A020F0")

        # Vẽ polyline theo 'points' (nếu model trả về)
        pts = p.get("points")
        flat_pts = extract_poly_points(pts) if pts is not None else []
        if len(flat_pts) >= 2:
            draw.line(flat_pts, fill="#A020F0", width=2)

    return overlay


def estimate_severity(p, img_w, img_h):
    """
    Ước lượng "mức độ nghiêm trọng" dựa trên diện tích box so với ảnh:
      - < 1%  : Nhỏ
      - 1–5%  : Trung bình
      - > 5%  : Lớn
    """
    w = float(p.get("width", 0))
    h = float(p.get("height", 0))
    if img_w <= 0 or img_h <= 0:
        return "Không xác định"

    area_box = w * h
    area_img = img_w * img_h
    ratio = area_box / area_img

    if ratio < 0.01:
        return "Nhỏ"
    elif ratio < 0.05:
        return "Trung bình"
    else:
        return "Lớn"


# =========================================================
# 3. GIAO DIỆN STREAMLIT
# =========================================================
st.set_page_config(page_title="BKAI - Crack Segmentation", layout="wide")

st.title("BKAI – Phát hiện & phân tích vết nứt bê tông bằng AI")

st.write(
    """
Tải một ảnh bê tông bất kỳ. Hệ thống sẽ:
- Kết luận: **Có vết nứt** hay **Không phát hiện vết nứt**
- Vẽ **khung + đường polyline** bao quanh vết nứt
- Hiển thị **biểu đồ độ tin cậy (confidence)** cho từng vết nứt
- Ước lượng **mức độ nghiêm trọng** dựa trên kích thước vùng nứt
"""
)

# Thanh bên
st.sidebar.header("Cấu hình")
min_conf = st.sidebar.slider(
    "Ngưỡng confidence tối thiểu để hiển thị (0–1)",
    0.0, 1.0, 0.3, 0.05
)
st.sidebar.caption("Chỉ vết nứt có độ tin cậy ≥ ngưỡng này mới được vẽ.")

# Form upload
with st.form("upload_form"):
    name = st.text_input("Họ tên (tùy chọn)")
    email = st.text_input("Email (tùy chọn)")
    note = st.text_area("Ghi chú về ảnh / công trình (tùy chọn)")
    uploaded_file = st.file_uploader("Chọn ảnh bê tông (JPG/PNG)", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Phân tích ảnh")


# =========================================================
# 4. XỬ LÝ KHI NGƯỜI DÙNG BẤM "PHÂN TÍCH ẢNH"
# =========================================================
if submitted:
    if uploaded_file is None:
        st.warning("Vui lòng chọn một ảnh trước khi bấm **Phân tích ảnh**.")
        st.stop()

    # Đọc ảnh
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Không đọc được ảnh: {e}")
        st.stop()

    img_w, img_h = image.size

    # Hai cột: ảnh gốc & ảnh kết quả
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_column_width=True)

    st.info("Đang gửi ảnh tới Roboflow…")

    # Chuẩn bị bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Gửi request
    try:
        resp = requests.post(
            ROBOFLOW_FULL_URL,
            files={"file": ("image.jpg", img_bytes, "image/jpeg")},
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi gọi API Roboflow: {e}")
        st.stop()

    if resp.status_code != 200:
        st.error("Roboflow trả lỗi. Hãy kiểm tra lại ROBOFLOW_FULL_URL (model_id, version, api_key).")
        st.write(f"Status code: {resp.status_code}")
        st.text(resp.text[:1500])
        st.stop()

    try:
        result = resp.json()
    except Exception as e:
        st.error(f"Không parse được JSON trả về: {e}")
        st.text(resp.text[:2000])
        st.stop()

    # ===== JSON raw (ẩn trong expander) =====
    with st.expander("Xem JSON raw (chi tiết kết quả từ model)", expanded=False):
        st.json(result)

    predictions = result.get("predictions", [])
    preds_conf = [p for p in predictions if float(p.get("confidence", 0)) >= min_conf]

    has_crack = len(predictions) > 0
    has_visible_crack = len(preds_conf) > 0

    # =====================================================
    # 4.1. ẢNH ĐÃ VẼ VẾT NỨT + KẾT LUẬN
    # =====================================================
    with col2:
        st.subheader("Kết quả trên ảnh")

        if not has_crack:
            st.image(image, use_column_width=True)
            st.success("✅ Kết luận: **Không phát hiện vết nứt** trong ảnh này.")
        elif not has_visible_crack:
            # Có crack nhưng tất cả dưới ngưỡng min_conf
            st.image(image, use_co_

