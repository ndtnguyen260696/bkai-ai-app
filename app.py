import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import time
import datetime
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.doctemplate import LayoutError

# =========================================================
# 0. CẤU HÌNH CHUNG
# =========================================================

# --- 0.1. Roboflow URL ---
ROBOFLOW_FULL_URL = (
    "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"
)

# --- 0.2. Logo BKAI (đặt cạnh app.py) ---
LOGO_PATH = "BKAI_Logo.png"

# --- 0.3. Font Unicode cho PDF ---
FONT_PATH = "times.ttf"   # nếu có Times New Roman thì đặt tên file này
FONT_NAME = "TimesVN"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
else:
    FONT_NAME = "DejaVu"
    pdfmetrics.registerFont(
        TTFont(FONT_NAME, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )

# --- 0.4. Cấu hình trang Streamlit ---
st.set_page_config(
    page_title="BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT BÊ TÔNG",
    layout="wide",
)

# =========================================================
# 1. HÀM XỬ LÝ ẢNH
# =========================================================


def extract_poly_points(points_field):
    """Chuyển 'points' trong JSON thành list [(x,y), ...]."""
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


def draw_predictions_with_mask(
    image: Image.Image, predictions, min_conf: float = 0.0
) -> Image.Image:
    """
    Vẽ ảnh phân tích với:
      - Box
      - Label
      - Vùng mask (polygon) màu xanh lá.
    """
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    green_solid = (0, 255, 0, 255)
    green_fill = (0, 255, 0, 80)

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

        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        # Box xanh lá
        draw.rectangle([x0, y0, x1, y1], outline=green_solid, width=3)

        # Nhãn trên mép box
        cls = p.get("class", "crack")
        label = f"{cls} {conf:.2f}"
        text_pos = (x0 + 3, y0 + 3)
        draw.text(text_pos, label, fill=green_solid)

        # Polyline + mask cùng màu xanh
        pts_raw = p.get("points")
        flat_pts = extract_poly_points(pts_raw) if pts_raw is not None else []
        if len(flat_pts) >= 3:
            draw.polygon(flat_pts, fill=green_fill)
            draw.line(flat_pts + [flat_pts[0]], fill=green_solid, width=3)

    result = Image.alpha_composite(base.convert("RGBA"), overlay)
    return result.convert("RGB")


def estimate_severity(p, img_w, img_h):
    """
    Ước lượng "mức độ nghiêm trọng" dựa trên diện tích box so với ảnh:
      - < 1%  : Nhỏ
      - 1–5%  : Trung bình
      - > 5%  : Nguy hiểm
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
        return "Nguy hiểm (Severe)"


# =========================================================
# 2. HÀM XUẤT PDF
# =========================================================


def export_pdf(original_img, analyzed_img, metrics_df, filename="bkai_report.pdf"):
    """Tạo file PDF báo cáo, đã hạn chế LayoutError."""

    left_margin = 25 * mm
    right_margin = 25 * mm
    top_margin = 20 * mm
    bottom_margin = 20 * mm

    page_w, page_h = A4
    content_width = page_w - left_margin - right_margin
    content_height = page_h - top_margin - bottom_margin

    def build_story(buf):
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=left_margin,
            rightMargin=right_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
        )

        styles = getSampleStyleSheet()
        for s in styles.byName:
            styles[s].fontName = FONT_NAME

        title_style = ParagraphStyle(
            "TitleVN",
            parent=styles["Title"],
            fontName=FONT_NAME,
            alignment=1,
            fontSize=20,
            leading=24,
        )
        h2 = ParagraphStyle(
            "Heading2VN",
            parent=styles["Heading2"],
            fontName=FONT_NAME,
            spaceBefore=12,
            spaceAfter=6,
        )
        normal = ParagraphStyle(
            "NormalVN",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            leading=13,
        )

        story = []

        from PIL import Image as PILImage

        def add_pil_image(pil_img, title_text):
            if pil_img is None:
                return
            if not isinstance(pil_img, PILImage.Image):
                pil_img = pil_img.convert("RGB")

            w, h = pil_img.size
            max_h = content_height * 0.4
            scale = min(content_width / w, max_h / h, 1.0)

            img_buf = io.BytesIO()
            pil_img.save(img_buf, format="PNG")
            img_buf.seek(0)

            story.append(Paragraph(title_text, h2))
            story.append(Spacer(1, 4 * mm))
            story.append(
                RLImage(
                    img_buf,
                    width=w * scale,
                    height=h * scale,
                )
            )
            story.append(Spacer(1, 6 * mm))

        # Logo + tiêu đề
        if os.path.exists(LOGO_PATH):
            story.append(RLImage(LOGO_PATH, width=40 * mm))
            story.append(Spacer(1, 6 * mm))

        story.append(Paragraph("BÁO CÁO KIỂM TRA VẾT NỨT BÊ TÔNG", title_style))
        story.append(Paragraph("Concrete Crack Inspection Report", normal))
        story.append(Spacer(1, 8 * mm))

        add_pil_image(original_img, "Ảnh gốc / Original Image")
        add_pil_image(analyzed_img, "Ảnh phân tích / Result Image")

        story.append(Paragraph("Bảng thông tin vết nứt / Crack Metrics", h2))

        data = [[
            Paragraph("Chỉ số (VI)", normal),
            Paragraph("Metric (EN)", normal),
            Paragraph("Giá trị / Value", normal),
            Paragraph("Ý nghĩa / Description", normal),
        ]]

        # Các dòng dữ liệu: dùng Paragraph để tự wrap + RÚT GỌN mô tả
        for _, row in metrics_df.iterrows():
            vi_txt = Paragraph(str(row["vi"]), normal)
            en_txt = Paragraph(str(row["en"]), normal)
            val_txt = Paragraph(str(row["value"]), normal)

            full_desc = str(row["desc"])
            if len(full_desc) > 180:
                short_desc = full_desc[:180] + "..."
            else:
                short_desc = full_desc
            desc_txt = Paragraph(short_desc, normal)

            data.append([vi_txt, en_txt, val_txt, desc_txt])

        col_widths = [
            0.2 * content_width,
            0.2 * content_width,
            0.2 * content_width,
            0.4 * content_width,
        ]

        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e88e5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 8 * mm))

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(
            Paragraph(
                f"BKAI © {datetime.datetime.now().year} – Report generated at {now_str}",
                normal,
            )
        )

        doc.build(story)

    buf = io.BytesIO()
    try:
        build_story(buf)
    except LayoutError:
        # Bản rút gọn nếu vẫn lỗi layout
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=left_margin,
            rightMargin=right_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
        )
        styles = getSampleStyleSheet()
        for s in styles.byName:
            styles[s].fontName = FONT_NAME
        normal = styles["Normal"]
        title_style = styles["Title"]
        story = []
        story.append(Paragraph("BKAI - Báo cáo rút gọn", title_style))
        story.append(Spacer(1, 10 * mm))
        story.append(
            Paragraph(
                "Nội dung chi tiết (ảnh hoặc bảng) quá lớn so với khổ giấy nên không thể hiển thị đầy đủ trong PDF. "
                "Vui lòng xem chi tiết trực tiếp trên giao diện web BKAI.",
                normal,
            )
        )
        doc.build(story)

    buf.seek(0)
    return buf


# =========================================================
# 3. STAGE 2 – DEMO KIẾN THỨC NỨT BÊ TÔNG
# =========================================================


def show_stage2_demo(key_prefix="stage2"):
    """Stage 2 demo: phân loại vết nứt & gợi ý nguyên nhân / biện pháp."""
    st.subheader("Stage 2 – Phân loại vết nứt & gợi ý nguyên nhân / biện pháp")

    options = [
        "Vết nứt dọc (Longitudinal Crack)",
        "Vết nứt ngang (Transverse Crack)",
        "Vết nứt mạng (Map Crack)",
    ]
    st.selectbox(
        "Chọn loại vết nứt (tóm tắt):",
        options,
        key=f"{key_prefix}_summary_selectbox",
    )

    demo_data = pd.DataFrame(
        [
            {
                "Loại vết nứt": "Vết nứt dọc (Longitudinal Crack)",
                "Nguyên nhân": "Co ngót, tải trọng trục bánh xe, bê tông chưa đủ cường độ.",
                "Biện pháp": "Kiểm tra khả năng chịu lực, gia cường hoặc trám vá bằng vật liệu phù hợp.",
            },
            {
                "Loại vết nứt": "Vết nứt ngang (Transverse Crack)",
                "Nguyên nhân": "Giãn nở nhiệt, không có khe co giãn, liên kết yếu.",
                "Biện pháp": "Tạo hoặc mở rộng khe co giãn, xử lý lại kết cấu nếu cần.",
            },
            {
                "Loại vết nứt": "Vết nứt mạng (Map Crack)",
                "Nguyên nhân": "Co ngót bề mặt, bê tông chất lượng thấp, bảo dưỡng kém.",
                "Biện pháp": "Loại bỏ lớp bề mặt yếu, phủ lớp vữa/bê tông mới có cường độ tốt hơn.",
            },
        ]
    )

    st.table(demo_data)
    st.caption("Stage 2 hiện tại là demo – bảng kiến thức cơ bản về các dạng vết nứt.")


# =========================================================
# 4. GIAO DIỆN CHÍNH (SAU KHI ĐĂNG NHẬP)
# =========================================================


def run_main_app():
    # Header với logo + tên user
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)
    with col_title:
        st.title("BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT BÊ TÔNG")
        user = st.session_state.get("username", "")
        if user:
            st.caption(f"Xin chào **{user}** – Phân biệt ảnh nứt / không nứt & xuất báo cáo.")
        else:
            st.caption("Phân biệt ảnh nứt / không nứt & xuất báo cáo.")

    st.write("---")

    st.sidebar.header("Cấu hình phân tích")
    min_conf = st.sidebar.slider(
        "Ngưỡng confidence tối thiểu",
        0.0,
        1.0,
        0.3,
        0.05,
    )
    st.sidebar.caption("Chỉ hiển thị những vết nứt có độ tin cậy ≥ ngưỡng này.")

    uploaded_files = st.file_uploader(
        "Tải một hoặc nhiều ảnh bê tông (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    analyze_btn = st.button("🔍 Phân tích ảnh")

    if analyze_btn:
        if not uploaded_files:
            st.warning("Vui lòng chọn ít nhất một ảnh trước khi bấm **Phân tích**.")
            st.stop()

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            st.write("---")
            st.markdown(f"## Ảnh {idx}: `{uploaded_file.name}`")

            t0 = time.time()
            orig_img = Image.open(uploaded_file).convert("RGB")
            img_w, img_h = orig_img.size

            buf = io.BytesIO()
            orig_img.save(buf, format="JPEG")
            buf.seek(0)

            with st.spinner(f"Đang gửi ảnh {idx} tới mô hình AI trên Roboflow..."):
                try:
                    resp
