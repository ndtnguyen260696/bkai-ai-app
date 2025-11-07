import time
import io
import datetime
import os

import streamlit as st
import requests
from PIL import Image, ImageDraw

# =========================================================
# 1. CẤU HÌNH ROBOFLOW
# =========================================================
# VÀO Roboflow: Project -> Deploy -> Hosted API -> Python
# COPY NGUYÊN URL DẠNG:
#   https://detect.roboflow.com/<model_id>/<version>?api_key=<API_KEY>
# DÁN VÀO GIỮA CẶP " " DƯỚI ĐÂY
ROBOFLOW_FULL_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"

#               ↑ THAY BẰNG URL CỦA BẠN (CHỈ 1 CẶP DẤU " ")


# =========================================================
# 2. CẤU HÌNH LOGO BKAI (LIÊN KẾT + TÙY CHỌN ẢNH)
# =========================================================

# Website chính mà bạn đưa:
BKAI_WEBSITE_URL = "https://bkai.b12sites.com/index"

# Nếu sau này bạn có ảnh logo (file local hoặc URL ảnh trực tiếp),
# hãy điền vào đây, ví dụ:
# BKAI_LOGO_IMAGE = "bkai_logo.png"  (file cùng thư mục với app.py)
# BKAI_LOGO_IMAGE = "https://.../logo.png"
BKAI_LOGO_IMAGE = ""  # hiện tại để rỗng -> chỉ hiển thị link website


def show_bkai_branding(max_width: int = 120):
    """
    Hiển thị brand BKAI một cách an toàn:
    - Nếu có BKAI_LOGO_IMAGE -> hiển thị ảnh
    - Luôn luôn có nút/link dẫn tới BKAI_WEBSITE_URL
    Không bao giờ để app crash vì logo.
    """
    try:
        if BKAI_LOGO_IMAGE:
            # Nếu là file local
            if os.path.exists(BKAI_LOGO_IMAGE):
                st.image(BKAI_LOGO_IMAGE, width=max_width)
            # Nếu là URL ảnh
            elif BKAI_LOGO_IMAGE.startswith("http"):
                st.image(BKAI_LOGO_IMAGE, width=max_width)
        # Nút/link tới website BKAI
        if BKAI_WEBSITE_URL:
            st.markdown(
                f"""
                <div style="text-align:center; padding-top:6px;">
                    <a href="{BKAI_WEBSITE_URL}" target="_blank" style="text-decoration:none;">
                        <span style="background-color:#1e293b; color:#e5e7eb;
                                     padding:4px 10px; border-radius:999px;
                                     font-size:13px;">
                            🌐 BKAI Website
                        </span>
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.warning(f"Không thể hiển thị logo/website BKAI ({e}).")


# =========================================================
# 3. HÀM HỖ TRỢ XỬ LÝ ẢNH
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


def draw_predictions(image: Image.Image, predictions, min_conf: float) -> Image.Image:
    """
    Vẽ Instance Segmentation:
      - Khung tím quanh vùng nứt (bounding box)
      - ĐƯỜNG & VÙNG segmentation theo 'points'
    """
    # Chuyển ảnh sang RGBA để hỗ trợ alpha (trong suốt)
    base = image.convert("RGBA")

    # Lớp để vẽ box + text
    box_draw = ImageDraw.Draw(base)

    # Lớp riêng để vẽ mask (tô màu vùng nứt)
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)

    # Màu tím
    purple_rgb = (160, 32, 240)        # #A020F0
    purple_rgba = (160, 32, 240, 255)
    purple_fill = (160, 32, 240, 80)   # tím trong suốt để tô vùng

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

        # Roboflow: x, y là tâm box
        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        # Khung tím quanh vùng nứt
        box_draw.rectangle([x0, y0, x1, y1], outline=purple_rgb, width=3)

        # Nhãn class + confidence
        cls = p.get("class", "crack")
        label = f"{cls} ({conf:.2f})"
        box_draw.text((x0 + 4, y0 + 4), label, fill=purple_rgb)

        # ===== VẼ INSTANCE SEGMENTATION TỪ 'points' =====
        pts = p.get("points")
        flat_pts = extract_poly_points(pts) if pts else []

        # Nếu model trả về đa giác (>= 3 điểm), tô vùng polygon
        if len(flat_pts) >= 3:
            mask_draw.polygon(flat_pts, outline=purple_rgba, fill=purple_fill)
        # Nếu chỉ có đường (>= 2 điểm) thì vẽ polyline
        elif len(flat_pts) >= 2:
            mask_draw.line(flat_pts, fill=purple_rgba, width=2)

    # Ghép lớp mask (tím trong suốt) lên ảnh gốc có box + text
    combined = Image.alpha_composite(base, mask_layer).convert("RGB")
    return combined


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


def resize_for_speed(image: Image.Image, max_side: int):
    """
    Giảm kích thước ảnh để xử lý nhanh hơn.
    Giữ nguyên tỉ lệ, cạnh dài nhất = max_side (nếu đang lớn hơn).
    """
    w, h = image.size
    max_current = max(w, h)
    if max_current <= max_side:
        return image, 1.0  # không thay đổi
    scale = max_side / max_current
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size), scale


# =========================================================
# 4. GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(page_title="BKAI - Crack Segmentation", layout="wide")

# CSS nền tối
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: logo + config
with st.sidebar:
    show_bkai_branding()
    st.markdown("### ⚙️ Cấu hình phân tích")
    min_conf = st.slider(
        "Ngưỡng confidence tối thiểu để hiển thị",
        0.0, 1.0, 0.3, 0.05,
    )
    max_side = st.slider(
        "Giới hạn kích thước cạnh dài nhất của ảnh (px)",
        400, 1600, 900, 100,
    )
    st.caption("Ảnh lớn sẽ được thu nhỏ về kích thước này để tăng tốc xử lý.")

# Header: cột logo + tiêu đề
col_logo, col_title = st.columns([1, 4])
with col_logo:
    show_bkai_branding(max_width=80)
with col_title:
    st.title("🧠 BKAI – Phát hiện & phân tích vết nứt bê tông")

st.markdown(
    """
Ứng dụng sử dụng **mô hình AI trên Roboflow** để:
- ✅ Kết luận: **Có vết nứt / Không phát hiện vết nứt**
- 🟣 Hiển thị **Instance Segmentation** (tô vùng nứt + đường polyline)
- 📊 Thống kê & **biểu đồ độ tin cậy (confidence)** cho từng vết nứt
"""
)

# Form upload
with st.form("upload_form"):
    name = st.text_input("Họ tên (tùy chọn)")
    email = st.text_input("Email (tùy chọn)")
    note = st.text_area("Ghi chú về ảnh / công trình (tùy chọn)")
    uploaded_file = st.file_uploader("📷 Chọn ảnh bê tông (JPG/PNG)", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("🚀 Phân tích ảnh")

# =========================================================
# 5. XỬ LÝ CHÍNH
# =========================================================
if submitted:
    if uploaded_file is None:
        st.warning("Vui lòng chọn một ảnh trước khi bấm **Phân tích ảnh**.")
        st.stop()

    # Đọc ảnh
    try:
        raw_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Không đọc được ảnh: {e}")
        st.stop()

    # Tối ưu kích thước
    image, scale = resize_for_speed(raw_image, max_side)
    img_w, img_h = image.size

    # Bố cục 2 cột: Ảnh gốc / Ảnh kết quả
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh gốc (đã tối ưu kích thước)")
        st.image(image, use_column_width=True)
        st.caption(f"Kích thước xử lý: {img_w} × {img_h} px (scale ~ {scale:.2f})")

    # Chuẩn bị bytes để gửi lên Roboflow
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Gọi API với spinner + đo thời gian
    with st.spinner("⏳ Đang gửi ảnh tới Roboflow và đợi mô hình phân tích..."):
        t0 = time.time()
        try:
            resp = requests.post(
                ROBOFLOW_FULL_URL,
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi khi gọi API Roboflow: {e}")
            st.stop()
        t1 = time.time()

    latency = t1 - t0

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

    predictions = result.get("predictions", [])
    preds_conf = [p for p in predictions if float(p.get("confidence", 0)) >= min_conf]
    has_crack = len(predictions) > 0
    has_visible_crack = len(preds_conf) > 0

    # ----- Ảnh đã vẽ kết quả + kết luận -----
    with col2:
        st.subheader("Ảnh đã đánh dấu vết nứt (Instance Segmentation)")
        if not has_crack:
            st.image(image, use_column_width=True)
            st.success("✅ Kết luận: **Không phát hiện vết nứt** trong ảnh này.")
        elif not has_visible_crack:
            st.image(image, use_column_width=True)
            st.info(
                f"Model có phát hiện vài tín hiệu yếu (confidence < {min_conf:.2f}), "
                "nhưng chưa đủ tin cậy theo ngưỡng bạn chọn."
            )
            st.warning("Kết luận: **Không có vết nứt rõ ràng** theo ngưỡng hiện tại.")
        else:
            annotated = draw_predictions(image, preds_conf, min_conf=min_conf)
            st.image(annotated, use_column_width=True)
            st.error("⚠️ Kết luận: **CÓ vết nứt** trong ảnh.")

    # ----- JSON raw (ẩn trong expander) -----
    with st.expander("📄 Xem JSON raw (dành cho kỹ thuật / nghiên cứu)", expanded=False):
        st.json(result)

    # =====================================================
    # 6. THỐNG KÊ + BIỂU ĐỒ
    # =====================================================
    st.write("---")
    st.subheader("📊 Thống kê và biểu đồ độ tin cậy")

    if not has_crack:
        st.write("🔍 Model không phát hiện vết nứt nào.")
    else:
        conf_all = [float(p.get("confidence", 0)) for p in predictions]
        max_conf = max(conf_all)
        min_conf_pred = min(conf_all)
        avg_conf = sum(conf_all) / len(conf_all)

        # 4 thẻ metric
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Số vùng nghi là vết nứt", len(predictions))
        mcol2.metric("Số vùng hiển thị", len(preds_conf))
        mcol3.metric("Confidence TB", f"{avg_conf:.2f}")
        mcol4.metric("Thời gian xử lý", f"{latency:.2f} s")

        # Bảng chi tiết
        rows = []
        for i, p in enumerate(predictions, start=1):
            conf = float(p.get("confidence", 0))
            sev = estimate_severity(p, img_w, img_h)
            rows.append(
                {
                    "Crack #": i,
                    "Confidence": round(conf, 3),
                    "Severity": sev,
                    "Width(px)": round(float(p.get("width", 0)), 1),
                    "Height(px)": round(float(p.get("height", 0)), 1),
                }
            )

        st.markdown("#### Bảng tóm tắt từng vết nứt")
        st.dataframe(rows, use_container_width=True)

        # Biểu đồ cột độ tin cậy
        st.markdown("#### Biểu đồ độ tin cậy của các vết nứt")
        chart_vals = [r["Confidence"] for r in rows]
        st.bar_chart(chart_vals)
        st.caption("Mỗi cột ứng với một vết nứt (Crack #1, #2, ...). Trục Y: confidence (0–1).")

    # ----- Thông tin phiên phân tích -----
    st.write("---")
    st.subheader("📝 Thông tin phiên phân tích")
    st.write(f"- Thời gian: **{datetime.datetime.now()}**")
    if name:
        st.write(f"- Người dùng: **{name}**")
    if email:
        st.write(f"- Email: {email}")
    if note:
        st.write(f"- Ghi chú: {note}")

