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
# 2. CẤU HÌNH LOGO BKAI
# =========================================================

BKAI_WEBSITE_URL = "https://bkai.b12sites.com/index"
# File logo (đặt cùng thư mục với app.py)
BKAI_LOGO_IMAGE = "bkai_logo.png"


def show_bkai_branding(max_width: int = 120):
    """
    Hiển thị brand BKAI:
    - Nếu có BKAI_LOGO_IMAGE -> hiển thị ảnh
    - Luôn luôn có nút/link dẫn tới BKAI_WEBSITE_URL
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
                <div style="text-align:center; padding-top:4px;">
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
    Vẽ kết quả:
      - Box xanh
      - Vùng nứt tô đỏ trong suốt (instance segmentation)
      - Nhãn 'crack 0.xx' trên nền xanh, chữ trắng
    """
    base = image.convert("RGBA")

    # Lớp vẽ box + label
    box_draw = ImageDraw.Draw(base)

    # Lớp riêng cho mask (tô vùng nứt)
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)

    # Màu xanh cho box
    blue_rgb = (0, 180, 255)
    # Màu đỏ cho mask
    red_rgba = (255, 0, 0, 255)
    red_fill = (255, 0, 0, 80)  # đỏ trong suốt

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

        # ===== VẼ MASK (INSTANCE SEGMENTATION) =====
        pts = p.get("points")
        flat_pts = extract_poly_points(pts) if pts else []

        if len(flat_pts) >= 3:
            mask_draw.polygon(flat_pts, fill=red_fill, outline=red_rgba)
        elif len(flat_pts) >= 2:
            mask_draw.line(flat_pts, fill=red_rgba, width=3)

        # ===== VẼ BOX XANH =====
        box_draw.rectangle([x0, y0, x1, y1], outline=blue_rgb, width=3)

        # ===== VẼ LABEL =====
        cls = p.get("class", "crack")
        label = f"{cls} {conf:.2f}"

        # Tính kích thước label
        try:
            text_bbox = box_draw.textbbox((0, 0), label)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            text_w, text_h = box_draw.textsize(label)

        # Label nằm trên mép trên box
        label_x0 = x0
        label_y1 = y0
        label_x1 = x0 + text_w + 6
        label_y0 = y0 - text_h - 6

        if label_y0 < 0:
            label_y0 = y0
            label_y1 = y0 + text_h + 6

        # nền label màu xanh
        box_draw.rectangle(
            [label_x0, label_y0, label_x1, label_y1],
            fill=blue_rgb
        )
        # chữ trắng
        box_draw.text(
            (label_x0 + 3, label_y0 + 3),
            label,
            fill="white"
        )

    # Ghép mask (đỏ trong suốt) lên ảnh gốc có box + label
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
        return image, 1.0
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
- 🟥 Hiển thị **Instance Segmentation**: vùng nứt đỏ trong suốt + box xanh
- 📊 Tạo **báo cáo tổng quan** cho từng ảnh
- 📈 Biểu đồ độ tin cậy cho từng vết nứt
- 📂 Hỗ trợ **phân tích nhiều ảnh cùng lúc**
"""
)

# Form upload (nhiều ảnh)
with st.form("upload_form"):
    name = st.text_input("Họ tên (tùy chọn)")
    email = st.text_input("Email (tùy chọn)")
    note = st.text_area("Ghi chú về ảnh / công trình (tùy chọn)")
    uploaded_files = st.file_uploader(
        "📷 Chọn 1 hoặc nhiều ảnh bê tông (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    submitted = st.form_submit_button("🚀 Phân tích ảnh")


# =========================================================
# 5. XỬ LÝ CHÍNH (CHO NHIỀU ẢNH)
# =========================================================
if submitted:
    if not uploaded_files:
        st.warning("Vui lòng chọn ít nhất một ảnh trước khi bấm **Phân tích ảnh**.")
    else:
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            st.write("---")
            st.markdown(f"## 🖼️ Ảnh {idx}: `{uploaded_file.name}`")

            # Đọc ảnh
            try:
                raw_image = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Không đọc được ảnh này: {e}")
                continue

            image, scale = resize_for_speed(raw_image, max_side)
            img_w, img_h = image.size

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ảnh gốc (đã tối ưu kích thước)")
                st.image(image, use_column_width=True)
                st.caption(f"Kích thước xử lý: {img_w} × {img_h} px (scale ~ {scale:.2f})")

            # Chuẩn bị bytes cho API
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            # Gọi API
            with st.spinner(f"⏳ Đang phân tích ảnh {idx}/{len(uploaded_files)} với Roboflow..."):
                t0 = time.time()
                try:
                    resp = requests.post(
                        ROBOFLOW_FULL_URL,
                        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                        timeout=60,
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"Lỗi khi gọi API Roboflow cho ảnh này: {e}")
                    continue
                t1 = time.time()

            latency = t1 - t0

            if resp.status_code != 200:
                st.error(
                    f"Roboflow trả lỗi với ảnh này (status {resp.status_code}). "
                    "Hãy kiểm tra lại ROBOFLOW_FULL_URL."
                )
                st.text(resp.text[:800])
                continue

            try:
                result = resp.json()
            except Exception as e:
                st.error(f"Không parse được JSON trả về cho ảnh này: {e}")
                st.text(resp.text[:800])
                continue

            predictions = result.get("predictions", [])
            preds_conf = [p for p in predictions if float(p.get("confidence", 0)) >= min_conf]
            has_crack = len(predictions) > 0
            has_visible_crack = len(preds_conf) > 0

            # ---- Kết luận để đưa vào bảng report ----
            if not has_crack:
                conclusion = "Không phát hiện vết nứt"
            elif not has_visible_crack:
                conclusion = "Không có vết nứt rõ ràng (dưới ngưỡng)"
            else:
                conclusion = "Có vết nứt"

            with col2:
                st.subheader("Ảnh đã đánh dấu vết nứt (mask đỏ + box xanh)")
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

            with st.expander("📄 Xem JSON raw cho ảnh này", expanded=False):
                st.json(result)

            # =================== BÁO CÁO & BIỂU ĐỒ ===================
            st.subheader("📊 Báo cáo tổng quan cho ảnh này")

            if not has_crack:
                st.write("🔍 Model không phát hiện vết nứt nào.")
            else:
                # Các thống kê độ tin cậy
                conf_all = [float(p.get("confidence", 0)) for p in predictions]
                max_conf = max(conf_all)
                min_conf_pred = min(conf_all)
                avg_conf = sum(conf_all) / len(conf_all)

                # Ước lượng mức độ nghiêm trọng lớn nhất
                severity_order = {"Nhỏ": 0, "Trung bình": 1, "Lớn": 2}
                max_severity = "Không xác định"
                if preds_conf:
                    for p in preds_conf:
                        sev = estimate_severity(p, img_w, img_h)
                        if max_severity not in severity_order or \
                           severity_order.get(sev, -1) > severity_order.get(max_severity, -1):
                            max_severity = sev

                # Độ phủ bề mặt vết nứt (dựa trên các vùng đạt ngưỡng)
                area_img = max(1, img_w * img_h)
                area_crack = 0.0
                for p in preds_conf:
                    w = float(p.get("width", 0))
                    h = float(p.get("height", 0))
                    area_crack += w * h
                coverage_ratio = area_crack / area_img
                coverage_percent = coverage_ratio * 100

                # Bảng báo cáo: 3 cột Chỉ số / Giá trị / Ghi chú
                report_rows = [
                    {"Chỉ số": "Kết luận chung",
                     "Giá trị": conclusion,
                     "Ghi chú": "Dựa trên số vùng đạt ngưỡng"},
                    {"Chỉ số": "Số vùng nghi là vết nứt",
                     "Giá trị": len(predictions),
                     "Ghi chú": "Tất cả predictions từ mô hình"},
                    {"Chỉ số": "Số vùng hiển thị theo ngưỡng",
                     "Giá trị": len(preds_conf),
                     "Ghi chú": f"Confidence ≥ {min_conf:.2f}"},
                    {"Chỉ số": "Độ tin cậy trung bình (pseudo-accuracy)",
                     "Giá trị": f"{avg_conf:.3f}",
                     "Ghi chú": "Trung bình confidence của tất cả vùng"},
                    {"Chỉ số": "Độ tin cậy cao nhất",
                     "Giá trị": f"{max_conf:.3f}",
                     "Ghi chú": ""},
                    {"Chỉ số": "Độ tin cậy thấp nhất",
                     "Giá trị": f"{min_conf_pred:.3f}",
                     "Ghi chú": ""},
                    {"Chỉ số": "Vết nứt nghiêm trọng nhất",
                     "Giá trị": max_severity,
                     "Ghi chú": "Ước lượng từ diện tích box so với ảnh"},
                    {"Chỉ số": "Độ phủ vết nứt trên bề mặt ảnh",
                     "Giá trị": f"{coverage_percent:.2f} %",
                     "Ghi chú": "Tổng diện tích các box đạt ngưỡng / diện tích ảnh"},
                    {"Chỉ số": "Thời gian suy luận",
                     "Giá trị": f"{latency:.2f} s",
                     "Ghi chú": "Thời gian gọi mô hình Roboflow"},
                    {"Chỉ số": "Kích thước ảnh xử lý",
                     "Giá trị": f"{img_w} × {img_h} px",
                     "Ghi chú": "Sau khi resize để tối ưu tốc độ"},
                    {"Chỉ số": "Ngưỡng confidence",
                     "Giá trị": f"{min_conf:.2f}",
                     "Ghi chú": ""},
                    {"Chỉ số": "F1-score",
                     "Giá trị": "N/A",
                     "Ghi chú": "Cần dữ liệu ground truth để tính"},
                    {"Chỉ số": "mAP",
                     "Giá trị": "N/A",
                     "Ghi chú": "Cần tập test chuẩn, không tính được từ 1 ảnh"},
                ]
                st.table(report_rows)
                st.caption(
                    "⚠️ Lưu ý: F1, mAP chỉ tính được khi có tập dữ liệu test có nhãn. "
                    "Ở đây chỉ hiển thị N/A mang tính tham khảo."
                )

                st.markdown("#### Chi tiết từng vết nứt trong ảnh này")
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

                st.dataframe(rows, use_container_width=True)

                st.markdown("#### Biểu đồ độ tin cậy các vết nứt trong ảnh này")
                chart_vals = [r["Confidence"] for r in rows]
                st.bar_chart(chart_vals)
                st.caption(
                    "Mỗi cột ứng với một vết nứt (Crack #1, #2, ...). Trục Y: confidence (0–1)."
                )

        # ======= Thông tin phiên phân tích chung =======
        st.write("---")
        st.subheader("📝 Thông tin phiên phân tích (chung cho tất cả ảnh)")
        st.write(f"- Thời gian: **{datetime.datetime.now()}**")
        if name:
            st.write(f"- Người dùng: **{name}**")
        if email:
            st.write(f"- Email: {email}")
        if note:
            st.write(f"- Ghi chú: {note}")
