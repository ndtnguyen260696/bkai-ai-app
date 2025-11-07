import os
import io
import time
import datetime
import requests
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

# =========================================================
# 🔧 1. CẤU HÌNH ROBOFLOW
# =========================================================
# Thay bằng API URL của bạn (copy nguyên dòng từ Roboflow → Deploy → Hosted API)
ROBOFLOW_FULL_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"

# =========================================================
# 🧩 2. LOGO BKAI
# =========================================================
BKAI_LOGO_PATH = "bkai_logo.png"  # đặt ảnh logo này trong cùng thư mục app.py
BKAI_WEBSITE_URL = "https://bkai.b12sites.com/index"

def show_bkai_logo(size: int = 120):
    """Hiển thị logo BKAI (nếu có file) hoặc fallback link website."""
    if os.path.exists(BKAI_LOGO_PATH):
        st.image(BKAI_LOGO_PATH, width=size)
    else:
        st.markdown(f"[🌐 BKAI Website]({BKAI_WEBSITE_URL})")

# =========================================================
# ⚙️ 3. HÀM HỖ TRỢ XỬ LÝ
# =========================================================
def extract_poly_points(points_field):
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
    """Vẽ vùng nứt đỏ trong suốt + box xanh + label dạng 'crack 0.85'."""
    base = image.convert("RGBA")
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)
    draw = ImageDraw.Draw(base)

    red_fill = (255, 0, 0, 80)
    red_outline = (255, 0, 0, 200)
    blue_box = (0, 180, 255)

    for p in predictions:
        conf = float(p.get("confidence", 0))
        if conf < min_conf:
            continue

        x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
        if None in (x, y, w, h):
            continue

        x0, y0, x1, y1 = x - w/2, y - h/2, x + w/2, y + h/2
        pts = extract_poly_points(p.get("points", []))

        if len(pts) >= 3:
            mask_draw.polygon(pts, fill=red_fill, outline=red_outline)
        elif len(pts) >= 2:
            mask_draw.line(pts, fill=red_outline, width=3)

        draw.rectangle([x0, y0, x1, y1], outline=blue_box, width=3)
        label = f"{p.get('class', 'crack')} {conf:.2f}"

        # Label nhỏ trên box
        try:
            text_bbox = draw.textbbox((0, 0), label)
            text_w, text_h = text_bbox[2]-text_bbox[0], text_bbox[3]-text_bbox[1]
        except Exception:
            text_w, text_h = draw.textsize(label)
        label_x1 = x0 + text_w + 6
        label_y0 = y0 - text_h - 6
        label_y1 = y0
        if label_y0 < 0:
            label_y0 = y0
            label_y1 = y0 + text_h + 6
        draw.rectangle([x0, label_y0, label_x1, label_y1], fill=blue_box)
        draw.text((x0+3, label_y0+2), label, fill="white")

    return Image.alpha_composite(base, mask_layer).convert("RGB")

def resize_for_speed(image: Image.Image, max_side: int):
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= max_side:
        return image, 1.0
    scale = max_side / max_dim
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size), scale

def estimate_severity(p, img_w, img_h):
    w, h = float(p.get("width", 0)), float(p.get("height", 0))
    ratio = (w * h) / (img_w * img_h)
    if ratio < 0.01:
        return "Nhỏ"
    elif ratio < 0.05:
        return "Trung bình"
    return "Lớn"

# =========================================================
# 🌙 4. GIAO DIỆN STREAMLIT
# =========================================================
st.set_page_config(page_title="BKAI Crack Detection", layout="wide")

# Theme nền tối nhẹ
st.markdown("""
<style>
body { background-color: #0f172a; color: #e2e8f0; }
table, th, td { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    show_bkai_logo(140)
    st.markdown("### ⚙️ Cấu hình mô hình")
    min_conf = st.slider("Ngưỡng confidence", 0.0, 1.0, 0.3, 0.05)
    max_side = st.slider("Kích thước tối đa của ảnh (px)", 400, 1600, 900, 100)
    st.caption("📏 Ảnh lớn sẽ được resize để tăng tốc độ xử lý.")

# Header
col_logo, col_title = st.columns([1, 4])
with col_logo:
    show_bkai_logo(80)
with col_title:
    st.title("🧠 BKAI – Phát hiện & phân tích vết nứt bê tông")

st.markdown(
    """
Ứng dụng này sử dụng **AI từ BKAI + Roboflow** để:
- 📸 Phân tích ảnh bê tông
- 🟥 Tô vùng nứt đỏ (Instance Segmentation)
- 🟦 Khoanh vùng bằng box xanh
- 📊 Tạo báo cáo tổng quan và chi tiết từng vết nứt
"""
)

# Upload nhiều ảnh
uploaded_files = st.file_uploader(
    "📂 Tải ảnh bê tông (JPG/PNG) – có thể chọn nhiều ảnh cùng lúc",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# =========================================================
# 🚀 5. XỬ LÝ ẢNH
# =========================================================
if uploaded_files:
    for idx, file in enumerate(uploaded_files, start=1):
        st.markdown(f"---\n## Ảnh {idx}: `{file.name}`")
        image = Image.open(file).convert("RGB")
        image, scale = resize_for_speed(image, max_side)
        w, h = image.size

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        with st.spinner("🔍 Đang gửi ảnh tới mô hình BKAI..."):
            t0 = time.time()
            resp = requests.post(
                ROBOFLOW_FULL_URL,
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                timeout=60,
            )
            t1 = time.time()

        latency = t1 - t0
        if resp.status_code != 200:
            st.error(f"Lỗi API ({resp.status_code})")
            st.text(resp.text)
            continue

        result = resp.json()
        preds = result.get("predictions", [])
        preds_conf = [p for p in preds if float(p.get("confidence", 0)) >= min_conf]

        # Vẽ kết quả
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", use_column_width=True)
        with col2:
            if preds_conf:
                annotated = draw_predictions(image, preds_conf, min_conf)
                st.image(annotated, caption="Ảnh kết quả (mask đỏ + box xanh)", use_column_width=True)
                st.error("⚠️ Kết luận: Có vết nứt được phát hiện.")
            else:
                st.image(image, caption="Không phát hiện vết nứt rõ ràng.", use_column_width=True)
                st.success("✅ Kết luận: Không có vết nứt đáng kể.")

        # Báo cáo tổng quan
        if preds:
            avg_conf = sum(float(p["confidence"]) for p in preds)/len(preds)
            max_conf = max(float(p["confidence"]) for p in preds)
            min_conf_pred = min(float(p["confidence"]) for p in preds)
            total_area = sum(float(p["width"])*float(p["height"]) for p in preds_conf)
            coverage = 100*total_area/(w*h)

            st.markdown("### 📊 Báo cáo tổng quan")
            df_summary = pd.DataFrame([
                ["Kết luận", "Có vết nứt" if preds_conf else "Không nứt"],
                ["Số vùng phát hiện", len(preds)],
                ["Số vùng đạt ngưỡng", len(preds_conf)],
                ["Độ tin cậy trung bình", f"{avg_conf:.2f}"],
                ["Độ tin cậy cao nhất", f"{max_conf:.2f}"],
                ["Độ tin cậy thấp nhất", f"{min_conf_pred:.2f}"],
                ["Độ phủ vết nứt", f"{coverage:.2f}%"],
                ["Thời gian xử lý", f"{latency:.2f}s"],
                ["Kích thước ảnh xử lý", f"{w} × {h}px"],
            ], columns=["Chỉ số", "Giá trị"])
            st.table(df_summary)

            # Chi tiết từng vết nứt
            st.markdown("### 🔎 Chi tiết từng vết nứt")
            detail_rows = []
            for i, p in enumerate(preds_conf, 1):
                sev = estimate_severity(p, w, h)
                detail_rows.append({
                    "Crack #": i,
                    "Confidence": round(float(p["confidence"]), 3),
                    "Mức độ": sev,
                    "Width(px)": round(float(p["width"]), 1),
                    "Height(px)": round(float(p["height"]), 1),
                })
            st.dataframe(detail_rows, use_container_width=True)

            # Biểu đồ độ tin cậy
            if detail_rows:
                st.markdown("### 📈 Biểu đồ độ tin cậy các vết nứt")
                conf_values = [r["Confidence"] for r in detail_rows]
                st.bar_chart(conf_values)

    st.markdown("---")
    st.caption("BKAI © 2025 – Ứng dụng phát hiện & phân tích vết nứt bê tông bằng AI.")
else:
    st.info("⬆️ Tải lên một hoặc nhiều ảnh để bắt đầu phân tích.")
