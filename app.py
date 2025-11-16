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
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.doctemplate import LayoutError
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# A4 xoay ngang cho Stage 2
A4_LANDSCAPE = landscape(A4)

# =========================================================
# 0. CẤU HÌNH CHUNG
# =========================================================

ROBOFLOW_FULL_URL = (
    "https://detect.roboflow.com/crack_segmentation_detection/4"
    "?api_key=nWA6ayjI5bGNpXkkbsAb"
)

LOGO_PATH = "BKAI_Logo.png"

FONT_PATH = "times.ttf"
FONT_NAME = "TimesVN"

# Cấu hình font PDF
if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
else:
    FONT_NAME = "DejaVuSans"
    try:
        pdfmetrics.registerFont(
            TTFont(FONT_NAME, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        )
    except Exception:
        pass

st.set_page_config(
    page_title="BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT",
    layout="wide",
)

# =========================================================
# 1. HÀM HỖ TRỢ CHUNG
# =========================================================

def fig_to_png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


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


def draw_predictions_with_mask(image: Image.Image, predictions, min_conf: float = 0.0):
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

        draw.rectangle([x0, y0, x1, y1], outline=green_solid, width=3)

        cls = p.get("class", "crack")
        label = f"{cls} {conf:.2f}"
        text_pos = (x0 + 3, y0 + 3)
        draw.text(text_pos, label, fill=green_solid)

        pts_raw = p.get("points")
        flat_pts = extract_poly_points(pts_raw) if pts_raw is not None else []
        if len(flat_pts) >= 3:
            draw.polygon(flat_pts, fill=green_fill)
            draw.line(flat_pts + [flat_pts[0]], fill=green_solid, width=3)

    result = Image.alpha_composite(base.convert("RGBA"), overlay)
    return result.convert("RGB")


def estimate_severity(p, img_w, img_h):
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
# 2. XUẤT PDF STAGE 1 – BẢN PRO (CÓ VẾT NỨT)
# =========================================================

def export_pdf(
    original_img,
    analyzed_img,
    metrics_df,
    chart_bar_png=None,
    chart_pie_png=None,
    filename="bkai_report_pro_plus.pdf",
):
    """
    BÁO CÁO BKAI – STAGE 1 (PRO, CÓ VẾT NỨT):
    - Dùng canvas, không Platypus.
    - Trang 1: logo + tiêu đề + 2 ảnh + banner kết luận + biểu đồ.
    - Trang 2+: bảng metrics.
    """

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    page_w, page_h = A4
    LEFT   = 20 * mm
    RIGHT  = 20 * mm
    TOP    = 20 * mm
    BOTTOM = 20 * mm
    CONTENT_W = page_w - LEFT - RIGHT

    TITLE_FONT      = FONT_NAME
    TITLE_SIZE      = 18
    BODY_FONT       = FONT_NAME
    BODY_SIZE       = 10
    SMALL_FONT_SIZE = 8

    # =================================================
    # HELPER: HEADER / FOOTER
    # =================================================
    def draw_header(page_title, subtitle=None, page_no=None):
        """
        Vẽ logo + tiêu đề, trả về y_top cho nội dung.
        """
        y_top = page_h - TOP

        # Logo
        logo_h = 0
        if os.path.exists(LOGO_PATH):
            try:
                logo = ImageReader(LOGO_PATH)
                logo_w = 30 * mm
                iw, ih = logo.getSize()
                logo_h = logo_w * ih / iw
                c.drawImage(
                    logo,
                    LEFT,
                    y_top - logo_h,
                    width=logo_w,
                    height=logo_h,
                    mask="auto",
                )
            except Exception:
                logo_h = 0

        # Tiêu đề
        c.setFillColor(colors.black)
        c.setFont(TITLE_FONT, TITLE_SIZE)
        c.drawCentredString(page_w / 2.0, y_top - 6 * mm, page_title)

        if subtitle:
            c.setFont(BODY_FONT, 11)
            c.drawCentredString(page_w / 2.0, y_top - 13 * mm, subtitle)

        # Footer
        footer_y = BOTTOM - 6
        c.setFont(BODY_FONT, SMALL_FONT_SIZE)
        c.setFillColor(colors.grey)
        footer = (
            f"BKAI – Concrete Crack Inspection | "
            f"Generated at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
        )
        c.drawString(LEFT, footer_y, footer)
        if page_no is not None:
            c.drawRightString(page_w - RIGHT, footer_y, f"Page {page_no}")

        # Nội dung bắt đầu cách logo khoảng 20mm
        content_start_y = y_top - max(logo_h, 15 * mm) - 20 * mm
        return content_start_y

    # =================================================
    # HELPER: VẼ ẢNH
    # =================================================
    def draw_pil_image(pil_img, x_left, top_y, max_w, max_h):
        if pil_img is None:
            return top_y
        img = ImageReader(pil_img)
        iw, ih = img.getSize()
        scale = min(max_w / iw, max_h / ih, 1.0)
        w = iw * scale
        h = ih * scale
        bottom_y = top_y - h
        c.drawImage(img, x_left, bottom_y, width=w, height=h, mask="auto")
        return bottom_y

    # =================================================
    # HELPER: WRAP TEXT
    # =================================================
    def wrap_text(text, font_name, font_size, max_width):
        words = str(text).split()
        if not words:
            return [""]

        lines = []
        current = words[0]
        for w in words[1:]:
            trial = current + " " + w
            w_trial = pdfmetrics.stringWidth(trial, font_name, font_size)
            if w_trial <= max_width:
                current = trial
            else:
                lines.append(current)
                current = w
        lines.append(current)
        return lines

    def draw_wrapped_cell(text, x_left, y_top, col_width, font_name, font_size, leading):
        inner_width = col_width - 4
        lines = wrap_text(text, font_name, font_size, inner_width)
        c.setFont(font_name, font_size)
        text_y = y_top - leading + 2
        for line in lines:
            c.drawString(x_left + 2, text_y, line)
            text_y -= leading
        used_height = leading * len(lines) + 4
        return used_height, len(lines)

    # =================================================
    # LẤY KẾT LUẬN & MỨC ĐỘ NGUY HIỂM
    # =================================================
    severity_val = ""
    summary_val = ""
    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            en = str(row.get("en", "")).strip()
            if en.lower() == "severity level":
                severity_val = str(row.get("value", ""))
            if en.lower() == "summary":
                summary_val = str(row.get("value", ""))

    if not summary_val:
        summary_val = "Kết luận: Ảnh bê tông có vết nứt, cần kiểm tra thêm."

    if "Nguy hiểm" in severity_val or "Severe" in severity_val:
        banner_fill = colors.HexColor("#ffebee")
        banner_text = colors.HexColor("#c62828")
    elif "Trung bình" in severity_val:
        banner_fill = colors.HexColor("#fff3e0")
        banner_text = colors.HexColor("#ef6c00")
    else:
        banner_fill = colors.HexColor("#e8f5e9")
        banner_text = colors.HexColor("#2e7d32")

    # =================================================
    # PAGE 1 – ẢNH + BIỂU ĐỒ
    # =================================================
    page_no = 1
    content_top_y = draw_header("BÁO CÁO KẾT QUẢ PHÂN TÍCH", page_no=page_no)

    # Hạ ảnh gốc & ảnh phân tích xuống thêm ~5mm
    content_top_y -= 5 * mm

    gap_x = 10 * mm
    slot_w = (CONTENT_W - gap_x) / 2.0
    max_img_h = 90 * mm

    c.setFont(BODY_FONT, 11)
    c.setFillColor(colors.black)
    c.drawString(LEFT, content_top_y + 4 * mm, "Ảnh gốc")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Ảnh phân tích")

    left_bottom = draw_pil_image(original_img, LEFT, content_top_y, slot_w, max_img_h)
    right_bottom = draw_pil_image(
        analyzed_img, LEFT + slot_w + gap_x, content_top_y, slot_w, max_img_h
    )
    images_bottom_y = min(left_bottom, right_bottom)

    # Banner kết luận
    banner_h = 16 * mm
    banner_bottom = images_bottom_y - 12 * mm
    if banner_bottom < BOTTOM + 40 * mm:
        banner_bottom = BOTTOM + 40 * mm

    c.setFillColor(banner_fill)
    c.setStrokeColor(colors.transparent)
    c.rect(LEFT, banner_bottom, CONTENT_W, banner_h, stroke=0, fill=1)

    c.setFillColor(banner_text)
    c.setFont(BODY_FONT, 11)
    c.drawString(LEFT + 4 * mm, banner_bottom + banner_h / 2.0 - 4, summary_val)

    # Biểu đồ
    charts_top_y = banner_bottom - 18 * mm
    max_chart_h = 70 * mm
    chart_slot_w = slot_w

    if chart_bar_png is not None:
        chart_bar_png.seek(0)
        bar_img = ImageReader(chart_bar_png)
        bw, bh = bar_img.getSize()
        scale_bar = min(chart_slot_w / bw, max_chart_h / bh)
        cw = bw * scale_bar
        ch = bh * scale_bar
        bar_bottom = charts_top_y - ch
        c.drawImage(bar_img, LEFT, bar_bottom, width=cw, height=ch, mask="auto")
        c.setFont(BODY_FONT, 10)
        c.setFillColor(colors.black)
        c.drawString(LEFT, bar_bottom - 10, "Độ tin cậy từng vùng nứt")

    if chart_pie_png is not None:
        chart_pie_png.seek(0)
        pie_img = ImageReader(chart_pie_png)
        pw, ph = pie_img.getSize()
        scale_pie = min(chart_slot_w / pw, max_chart_h / ph)
        cw = pw * scale_pie
        ch = ph * scale_pie
        pie_bottom = charts_top_y - ch
        c.drawImage(
            pie_img,
            LEFT + chart_slot_w + gap_x,
            pie_bottom,
            width=cw,
            height=ch,
            mask="auto",
        )
        c.setFont(BODY_FONT, 10)
        c.setFillColor(colors.black)
        c.drawString(
            LEFT + chart_slot_w + gap_x,
            pie_bottom - 10,
            "Tỷ lệ vùng nứt so với toàn ảnh",
        )

    c.showPage()

    # =================================================
    # PAGE 2+ – BẢNG METRICS
    # =================================================
    page_no += 1
    subtitle = "Bảng tóm tắt các chỉ số vết nứt"
    content_top_y = draw_header(
        "BÁO CÁO KẾT QUẢ PHÂN TÍCH", subtitle=subtitle, page_no=page_no
    )

    rows = []
    skip_keys = {"Crack Length", "Crack Width"}
    for _, r in metrics_df.iterrows():
        en_name = str(r.get("en", "")).strip()
        if en_name in skip_keys:
            continue
        label = f"{r.get('vi', '')} ({en_name})"
        val = str(r.get("value", ""))
        rows.append((label, val))

    if not rows:
        c.save()
        buf.seek(0)
        return buf

    col1_w = 12 * mm
    col2_w = 95 * mm
    col3_w = CONTENT_W - col1_w - col2_w

    header_h   = 10 * mm
    base_lead  = 4.0
    max_body_y = content_top_y - 10 * mm

    def start_table_page(page_no):
        c.showPage()
        y0 = draw_header(
            "BÁO CÁO KẾT QUẢ PHÂN TÍCH", subtitle=subtitle, page_no=page_no
        )
        return y0 - 10 * mm

    table_top_y = max_body_y
    x0 = LEFT
    x1 = x0 + col1_w
    x2 = x1 + col2_w

    def draw_table_header(top_y):
        c.setFillColor(colors.HexColor("#1e88e5"))
        c.rect(x0, top_y - header_h, CONTENT_W, header_h, stroke=0, fill=1)
        c.setFont(BODY_FONT, 10)
        c.setFillColor(colors.white)
        c.drawString(x0 + 2, top_y - header_h + 3, "No.")
        c.drawString(x1 + 2, top_y - header_h + 3, "Chỉ số (VI / EN)")
        c.drawString(x2 + 2, top_y - header_h + 3, "Giá trị / Value")
        return top_y - header_h

    current_y = draw_table_header(table_top_y)

    for i, (label, val) in enumerate(rows, start=1):
        label_lines = wrap_text(label, BODY_FONT, BODY_SIZE, col2_w - 4)
        value_lines = wrap_text(val, BODY_FONT, BODY_SIZE, col3_w - 4)
        n_lines = max(len(label_lines), len(value_lines))
        leading = BODY_SIZE + base_lead
        row_h = n_lines * leading + 6

        if current_y - row_h < BOTTOM + 30 * mm:
            page_no += 1
            current_y = start_table_page(page_no)
            current_y = draw_table_header(current_y)

        if i % 2 == 0:
            c.setFillColor(colors.HexColor("#e3f2fd"))
            c.rect(x0, current_y - row_h, CONTENT_W, row_h, stroke=0, fill=1)

        c.setStrokeColor(colors.grey)
        c.setLineWidth(0.3)
        c.rect(x0, current_y - row_h, CONTENT_W, row_h, stroke=1, fill=0)

        c.setFont(BODY_FONT, BODY_SIZE)
        c.setFillColor(colors.black)
        c.drawString(x0 + 2, current_y - leading, str(i))

        draw_wrapped_cell(label, x1, current_y, col2_w, BODY_FONT, BODY_SIZE, leading)
        draw_wrapped_cell(val,   x2, current_y, col3_w, BODY_FONT, BODY_SIZE, leading)

        current_y -= row_h

    c.save()
    buf.seek(0)
    return buf

# =========================================================
# PDF CHO TRƯỜNG HỢP KHÔNG CÓ VẾT NỨT
# =========================================================

def export_pdf_no_crack(original_img):
    """
    Báo cáo 1 trang khi KHÔNG phát hiện vết nứt:
    - Logo + tiêu đề
    - Ảnh gốc + Ảnh phân tích (cùng là ảnh gốc)
    - Dòng kết luận bên dưới
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    page_w, page_h = A4
    LEFT   = 20 * mm
    RIGHT  = 20 * mm
    TOP    = 20 * mm
    BOTTOM = 20 * mm
    CONTENT_W = page_w - LEFT - RIGHT

    TITLE_FONT = FONT_NAME
    BODY_FONT  = FONT_NAME

    def draw_header_no_crack():
        y_top = page_h - TOP

        logo_h = 0
        if os.path.exists(LOGO_PATH):
            try:
                logo = ImageReader(LOGO_PATH)
                logo_w = 30 * mm
                iw, ih = logo.getSize()
                logo_h = logo_w * ih / iw
                c.drawImage(
                    logo,
                    LEFT,
                    y_top - logo_h,
                    width=logo_w,
                    height=logo_h,
                    mask="auto",
                )
            except Exception:
                logo_h = 0

        c.setFont(TITLE_FONT, 18)
        c.drawCentredString(page_w / 2, y_top - 6 * mm, "BÁO CÁO KẾT QUẢ PHÂN TÍCH")
        c.setFont(BODY_FONT, 11)
        c.drawCentredString(
            page_w / 2,
            y_top - 14 * mm,
            "Trường hợp: Không phát hiện vết nứt rõ ràng",
        )

        content_top = y_top - max(logo_h, 15 * mm) - 20 * mm
        return content_top

    content_top_y = draw_header_no_crack()

    # Ảnh gốc & Ảnh phân tích
    max_img_h = 90 * mm
    gap_x = 10 * mm
    slot_w = (CONTENT_W - gap_x) / 2

    def draw_pil(img, x, top):
        ir = ImageReader(img)
        iw, ih = ir.getSize()
        scale = min(slot_w / iw, max_img_h / ih, 1.0)
        w = iw * scale
        h = ih * scale
        bottom = top - h
        c.drawImage(ir, x, bottom, width=w, height=h, mask="auto")
        return bottom

    c.setFont(BODY_FONT, 11)
    c.drawString(LEFT, content_top_y + 4 * mm, "Ảnh gốc")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Ảnh phân tích")

    left_bottom = draw_pil(original_img, LEFT, content_top_y)
    _           = draw_pil(original_img, LEFT + slot_w + gap_x, content_top_y)

    # Kết luận
    banner_y = left_bottom - 12 * mm
    banner_h = 16 * mm

    c.setFillColor(colors.HexColor("#e8f5e9"))
    c.rect(LEFT, banner_y, CONTENT_W, banner_h, stroke=0, fill=1)

    c.setFillColor(colors.HexColor("#2e7d32"))
    c.setFont(BODY_FONT, 11)
    c.drawString(
        LEFT + 4 * mm,
        banner_y + banner_h / 2 - 4,
        "Không phát hiện vết nứt rõ ràng trong ảnh theo ngưỡng của mô hình.",
    )

    # Footer đơn giản
    footer_y = BOTTOM - 6
    c.setFont(BODY_FONT, 8)
    c.setFillColor(colors.grey)
    c.drawString(
        LEFT,
        footer_y,
        f"BKAI – Concrete Crack Inspection | Generated at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
    )
    c.drawRightString(page_w - RIGHT, footer_y, "Page 1")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# =========================================================
# 3. XUẤT PDF STAGE 2 (KIẾN THỨC, LANDSCAPE)
# =========================================================

def export_stage2_pdf(component_df: pd.DataFrame) -> io.BytesIO:
    """
    Xuất PDF KIẾN THỨC STAGE 2:
    - Logo BKAI + tiêu đề giống Stage 1.
    - Bảng 5 cột có hình minh hoạ.
    - A4 xoay ngang để bảng không tràn.
    """

    left_margin   = 20 * mm
    right_margin  = 20 * mm
    top_margin    = 20 * mm
    bottom_margin = 20 * mm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4_LANDSCAPE,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )

    page_w, page_h = A4_LANDSCAPE
    usable_width = page_w - left_margin - right_margin

    styles = getSampleStyleSheet()
    for s in styles.byName:
        styles[s].fontName = FONT_NAME

    title_style = ParagraphStyle(
        "TitleStage2",
        parent=styles["Title"],
        fontName=FONT_NAME,
        alignment=1,
        fontSize=18,
        leading=22,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "SubTitleStage2",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        alignment=1,
        fontSize=10,
        leading=12,
        textColor=colors.grey,
        spaceAfter=8,
    )
    normal = ParagraphStyle(
        "NormalStage2",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=8,
        leading=10,
    )

    elements = []

    # Header: logo + title
    header_row = []
    if os.path.exists(LOGO_PATH):
        logo_flow = RLImage(LOGO_PATH, width=28 * mm, height=28 * mm)
        header_row.append(logo_flow)
        header_row.append(
            Paragraph("BKAI – BÁO CÁO KIẾN THỨC VẾT NỨT (STAGE 2)", title_style)
        )
        header_table = Table(
            [header_row],
            colWidths=[30 * mm, doc.width - 30 * mm],
            hAlign="LEFT",
        )
        header_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("GRID", (0, 0), (-1, -1), 0, colors.white),
                ]
            )
        )
        elements.append(header_table)
    else:
        elements.append(
            Paragraph("BKAI – BÁO CÁO KIẾN THỨC VẾT NỨT (STAGE 2)", title_style)
        )

    elements.append(
        Paragraph(
            "Bảng phân loại các vết nứt bê tông thường gặp theo từng loại cấu kiện (dầm, cột, sàn, tường).",
            subtitle_style,
        )
    )

    # Chuẩn bị dữ liệu bảng
    data = [
        [
            Paragraph("Cấu kiện", normal),
            Paragraph("Loại vết nứt", normal),
            Paragraph("Nguyên nhân hình thành vết nứt", normal),
            Paragraph("Đặc trưng về hình dạng vết nứt", normal),
            Paragraph("Hình ảnh minh họa vết nứt", normal),
        ]
    ]

    def make_thumb(path: str):
        if isinstance(path, str) and path and os.path.exists(path):
            return RLImage(path, width=25 * mm, height=25 * mm)
        else:
            return Paragraph("—", normal)

    for _, row in component_df.iterrows():
        img_path = row.get("Ảnh (path)", "") or row.get("Hình ảnh minh họa", "")
        data.append(
            [
                Paragraph(str(row["Cấu kiện"]), normal),
                Paragraph(str(row["Loại vết nứt"]), normal),
                Paragraph(str(row["Nguyên nhân"]), normal),
                Paragraph(str(row["Đặc trưng hình dạng"]), normal),
                make_thumb(img_path),
            ]
        )

    table = Table(
        data,
        colWidths=[
            0.12 * usable_width,  # Cấu kiện
            0.18 * usable_width,  # Loại vết nứt
            0.30 * usable_width,  # Nguyên nhân
            0.25 * usable_width,  # Đặc trưng
            0.15 * usable_width,  # Ảnh
        ],
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e88e5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, 0), 9),

                ("FONTNAME", (0, 1), (-2, -1), FONT_NAME),
                ("FONTSIZE", (0, 1), (-2, -1), 8),
                ("VALIGN", (0, 1), (-1, -1), "TOP"),
                ("ALIGN", (0, 1), (-2, -1), "LEFT"),
                ("ALIGN", (-1, 1), (-1, -1), "CENTER"),

                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),

                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )

    elements.append(table)

    doc.build(elements)
    buf.seek(0)
    return buf

# =========================================================
# 4. STAGE 2 – TABLE ĐẸP + MAPPING ẢNH (STREAMLIT)
# =========================================================

def render_component_crack_table(component_df: pd.DataFrame):
    st.markdown("### 2.2. Bảng chi tiết vết nứt theo cấu kiện")

    h1, h2, h3, h4, h5 = st.columns([1, 1.2, 2.2, 2.2, 1.6])
    header_style = (
        "background-color:#e3f2fd;padding:6px;border:1px solid #90caf9;"
        "font-weight:bold;text-align:center;"
    )
    h1.markdown(f"<div style='{header_style}'>Cấu kiện</div>", unsafe_allow_html=True)
    h2.markdown(f"<div style='{header_style}'>Loại vết nứt</div>", unsafe_allow_html=True)
    h3.markdown(
        f"<div style='{header_style}'>Nguyên nhân hình thành vết nứt</div>",
        unsafe_allow_html=True,
    )
    h4.markdown(
        f"<div style='{header_style}'>Đặc trưng về hình dạng vết nứt</div>",
        unsafe_allow_html=True,
    )
    h5.markdown(
        f"<div style='{header_style}'>Hình ảnh minh họa vết nứt</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin:2px 0 6px 0;'>", unsafe_allow_html=True)

    for component, subdf in component_df.groupby("Cấu kiện"):
        st.markdown(
            f"<div style='background-color:#bbdefb;padding:4px 10px;margin:4px 0;"
            f"font-weight:bold;border-left:4px solid #1976d2;'>"
            f"{component.upper()}</div>",
            unsafe_allow_html=True,
        )

        first_row = True
        for _, row in subdf.iterrows():
            c1, c2, c3, c4, c5 = st.columns([1, 1.2, 2.2, 2.2, 1.6])

            if first_row:
                c1.markdown(
                    f"<div style='padding:4px;font-weight:bold;'>{component}</div>",
                    unsafe_allow_html=True,
                )
                first_row = False
            else:
                c1.markdown("&nbsp;", unsafe_allow_html=True)

            c2.write(row["Loại vết nứt"])
            c3.write(row["Nguyên nhân"])
            c4.write(row["Đặc trưng hình dạng"])

            img_path = row.get("Ảnh (path)", "") or row.get("Hình ảnh minh họa", "")
            if isinstance(img_path, str) and img_path and os.path.exists(img_path):
                c5.image(img_path, use_container_width=True)
            else:
                c5.write("—")

        st.markdown(
            "<hr style='margin:6px 0 10px 0;border-top:1px dashed #b0bec5;'>",
            unsafe_allow_html=True,
        )

def show_stage2_demo(key_prefix="stage2"):
    st.subheader("Stage 2 – Phân loại vết nứt & gợi ý nguyên nhân / biện pháp")

    # 2.0 Hình minh hoạ
    st.markdown("### 2.0. Sơ đồ & ví dụ vết nứt trên kết cấu")
    col_img1, col_img2 = st.columns([3, 4])
    with col_img1:
        tree_path = "images/stage2_crack_tree.png"
        if os.path.exists(tree_path):
            st.image(
                tree_path,
                caption=(
                    "Sơ đồ phân loại các loại vết nứt theo thời điểm xuất hiện "
                    "và mức độ ảnh hưởng"
                ),
                use_container_width=True,
            )
        else:
            st.info("Chưa thấy images/stage2_crack_tree.png")
    with col_img2:
        example_path = "images/stage2_structural_example.png"
        if os.path.exists(example_path):
            st.image(
                example_path,
                caption="Ví dụ các loại vết nứt kết cấu bê tông (dầm, cột, tường, sàn)",
                use_container_width=True,
            )
        else:
            st.info("Chưa thấy images/stage2_structural_example.png")

    st.markdown("---")

    # 2.1 Bảng tổng hợp theo cơ chế (giữ cho phụ lục)
    options = [
        "I.1 Nứt co ngót dẻo (Plastic Shrinkage Crack)",
        "I.2 Nứt lún dẻo / lắng dẻo (Plastic Settlement Crack)",
        "II.1 Nứt do co ngót khô (Drying Shrinkage Crack)",
        "II.2 Nứt do đóng băng – băng tan (Freeze–Thaw Crack)",
        "II.3 Nứt do nhiệt (Thermal Crack)",
        "II.4a Nứt do hoá chất – sunfat tấn công (Sulfate Attack)",
        "II.4b Nứt do hoá chất – kiềm cốt liệu (Alkali–Aggregate Reaction)",
        "II.5 Nứt do ăn mòn cốt thép (Corrosion–Induced Crack)",
        "II.6a Nứt do tải trọng – nứt uốn (Flexural Crack)",
        "II.6b Nứt do tải trọng – nứt cắt/nén/xoắn (Shear/Compression/Torsion Cracks)",
        "II.7 Nứt do lún (Settlement Crack)",
    ]
    st.selectbox(
        "Chọn loại vết nứt (tóm tắt):",
        options,
        key=f"{key_prefix}_summary_selectbox",
    )

    st.caption(
        "Bảng 1 – Tổng hợp các dạng nứt theo cơ chế hình thành và biện pháp kiểm soát "
        "(có thể dùng làm phụ lục trong luận văn)."
    )

    # 2.2 Bảng 2 – mapping ảnh đầy đủ
    st.subheader("Phân loại các vết nứt bê tông thường xảy ra cho từng loại cấu kiện")

    component_crack_data = pd.DataFrame(
        [
            # ===== DẦM =====
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt uốn",
                "Nguyên nhân": (
                    "Do mô men uốn vượt quá giới hạn chịu tải của dầm; "
                    "tiết diện hoặc cốt thép chịu uốn không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt thường chéo hoặc hơi cong, xuất hiện nhiều ở giữa nhịp; "
                    "rộng nhất ở vùng chịu kéo."
                ),
                "Ảnh (path)": "images/stage2/beam_uon.png",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt cắt",
                "Nguyên nhân": (
                    "Lực cắt lớn tại gối hoặc gần điểm uốn; khả năng chịu cắt của bê tông/cốt đai không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt xiên khoảng 45° so với trục dầm; có thể đơn lẻ hoặc nhóm."
                ),
                "Ảnh (path)": "images/stage2/beam_cat.png",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt xoắn",
                "Nguyên nhân": (
                    "Độ bền xoắn không đủ; thiếu cốt thép chịu xoắn; tiết diện dầm không phù hợp."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chéo, dạng xoắn ốc hoặc ziczac quanh dầm; "
                    "bề rộng tương đối đồng đều."
                ),
                "Ảnh (path)": "images/stage2/beam_xoan.png",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt trượt",
                "Nguyên nhân": (
                    "Bê tông bị xáo trộn khi cường độ chưa đạt; gối đỡ/cốp pha dịch chuyển."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt gần mép gối đỡ, chạy gần phương thẳng đứng; "
                    "rộng nhất tại đáy dầm."
                ),
                "Ảnh (path)": "images/stage2/beam_truot.png",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt kéo",
                "Nguyên nhân": (
                    "Cốt thép chịu kéo không đủ, dầm quá tải, biến dạng không đều."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt gần vuông góc với trục dầm; phía dưới rộng hơn phía trên; "
                    "thường song song."
                ),
                "Ảnh (path)": "images/stage2/beam_keo.png",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt ăn mòn cốt thép",
                "Nguyên nhân": (
                    "Lớp bảo vệ mỏng, môi trường xâm thực; cốt thép gỉ giãn nở ép vào bê tông."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chạy dọc theo thanh thép; thường kèm hoen gỉ, bong lớp bảo vệ."
                ),
                "Ảnh (path)": "images/stage2/beam_anmon.png",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt co ngót",
                "Nguyên nhân": (
                    "Bê tông co ngót do mất nước, bị kiềm chế bởi cốt thép/kết cấu lân cận."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt nhỏ, nhiều, có thể vuông góc trục dầm hoặc tạo mạng lưới."
                ),
                "Ảnh (path)": "images/stage2/beam_congot.png",
            },

            # ===== CỘT =====
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt ngang",
                "Nguyên nhân": (
                    "Không đủ mô-men kiềm chế, diện tích cốt thép nhỏ; chịu uốn/cắt lớn."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt ngang quanh cột, thường tại vùng nối dầm–cột."
                ),
                "Ảnh (path)": "images/stage2/column_ngang.png",
            },
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt chéo",
                "Nguyên nhân": (
                    "Cột chịu nén – uốn / cắt lớn; thiết kế hoặc cường độ vật liệu không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt xiên trên bề mặt cột, xuất hiện khi tải gần/vượt sức chịu tải."
                ),
                "Ảnh (path)": "images/stage2/column_cheo.png",
            },
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt tách (dọc)",
                "Nguyên nhân": (
                    "Cốt thép dọc không đủ; bê tông cường độ thấp; ứng suất nén lớn gây tách dọc."
                ),
                "Đặc trưng hình dạng": (
                    "Các vết nứt dọc song song, độ dài và rộng khác nhau."
                ),
                "Ảnh (path)": "images/stage2/column_tach.png",
            },
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt do ăn mòn",
                "Nguyên nhân": (
                    "Cốt thép bị gỉ do môi trường xâm thực; sản phẩm ăn mòn giãn nở."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt dọc theo cốt thép; bong tróc, vết gỉ trên bề mặt."
                ),
                "Ảnh (path)": "images/stage2/column_anmon.png",
            },
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt co ngót",
                "Nguyên nhân": (
                    "Co ngót bê tông bị kiềm chế bởi cốt thép và cấu kiện liên kết."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt dọc mảnh, nhiều, phân bố tương đối đều."
                ),
                "Ảnh (path)": "images/stage2/column_congot.png",
            },

            # ===== SÀN =====
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt co ngót dẻo",
                "Nguyên nhân": (
                    "Nhiệt độ cao, gió, độ ẩm thấp; bốc hơi nước nhanh khi bê tông còn dẻo."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt nông, nhỏ; hình dạng ngẫu nhiên, đa giác."
                ),
                "Ảnh (path)": "images/stage2/slab_congot_deo.png",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt co ngót khô",
                "Nguyên nhân": (
                    "Co ngót do nước bay hơi sau khi bê tông đông cứng trong môi trường khô/nóng."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt rõ, tạo mạng lưới (map cracking) hoặc đường thẳng."
                ),
                "Ảnh (path)": "images/stage2/slab_congot_kho.png",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt do nhiệt",
                "Nguyên nhân": "Chênh lệch nhiệt độ giữa bề mặt và bên trong sàn.",
                "Đặc trưng hình dạng": (
                    "Vết nứt bề mặt, có thể kết hợp bong tróc lớp bê tông."
                ),
                "Ảnh (path)": "images/stage2/slab_nhiet.png",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt uốn",
                "Nguyên nhân": (
                    "Mô men uốn vượt khả năng chịu uốn; thép chịu kéo không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chéo/hơi cong, rộng nhất ở mặt chịu kéo (thường mặt dưới giữa nhịp)."
                ),
                "Ảnh (path)": "images/stage2/slab_uon.png",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt cắt",
                "Nguyên nhân": (
                    "Lực cắt lớn gần gối hoặc vùng chịu tải tập trung; thiếu thép chịu cắt."
                ),
                "Đặc trưng hình dạng": "Vết nứt xiên ~45° so với trục sàn.",
                "Ảnh (path)": "images/stage2/slab_cat.png",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt xoắn",
                "Nguyên nhân": (
                    "Sàn làm việc như bản chịu xoắn (bản console, vùng góc…); độ bền xoắn không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chéo dạng xoắn ốc; bề rộng tương đối đồng đều."
                ),
                "Ảnh (path)": "images/stage2/slab_xoan.png",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt ăn mòn cốt thép",
                "Nguyên nhân": (
                    "Ion Cl-, nước biển, muối khử băng xâm nhập; lớp bảo vệ mỏng; thép gỉ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chạy dọc theo thép; kèm hoen gỉ, bong lớp bảo vệ."
                ),
                "Ảnh (path)": "images/stage2/slab_anmon.png",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt do tải trọng – lực tập trung",
                "Nguyên nhân": "Quá tải cục bộ; thiếu cốt thép chịu uốn cục bộ.",
                "Đặc trưng hình dạng": (
                    "Vết nứt vuông góc phương ứng suất kéo; dạng chữ thập/tỏa ra từ điểm tải."
                ),
                "Ảnh (path)": "images/stage2/slab_taptrung.png",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt do tải trọng – lực phân bố",
                "Nguyên nhân": (
                    "Tải phân bố vượt khả năng làm việc lâu dài; sàn thiếu độ cứng."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt dạng chữ thập, mạng lưới hoặc xiên từ giữa sàn ra cạnh."
                ),
                "Ảnh (path)": "images/stage2/slab_phanbo.png",
            },

            # ===== TƯỜNG =====
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt co ngót",
                "Nguyên nhân": (
                    "Bề mặt tường bốc hơi nước nhanh; ứng suất co ngót vượt khả năng chịu kéo."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt bề mặt ngẫu nhiên, đa giác, bắt chéo hoặc song song."
                ),
                "Ảnh (path)": "images/stage2/wall_congot.png",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt do nhiệt",
                "Nguyên nhân": (
                    "Chênh lệch nhiệt độ trong bề dày tường; giãn nở/co lại không đều."
                ),
                "Đặc trưng hình dạng": (
                    "Thường là vết nứt thẳng đứng; rộng hơn ở vùng chịu kéo do nhiệt."
                ),
                "Ảnh (path)": "images/stage2/wall_nhiet.png",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt ngang do tải trọng",
                "Nguyên nhân": (
                    "Tường chịu tải vượt mức; phân bố tải không đều; trượt/xoay tại chân tường."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt ngang chia tường thành hai phần; phần trên có thể nghiêng."
                ),
                "Ảnh (path)": "images/stage2/wall_ngang_taitrong.png",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt dọc do tải trọng",
                "Nguyên nhân": "Tải đứng lớn, lún cục bộ, thiếu thép dọc.",
                "Đặc trưng hình dạng": (
                    "Vết nứt tách dọc chia tường thành hai mảng song song."
                ),
                "Ảnh (path)": "images/stage2/wall_doc_taitrong.png",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt chéo do tải trọng",
                "Nguyên nhân": (
                    "Tường vừa chịu nén vừa chịu cắt/uốn do tải ngang và đứng."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chéo; rộng nhất gần vùng chịu lực lớn."
                ),
                "Ảnh (path)": "images/stage2/wall_cheo_taitrong.png",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt ăn mòn cốt thép",
                "Nguyên nhân": (
                    "Cốt thép tường bị gỉ; sản phẩm ăn mòn giãn nở làm nứt lớp bảo vệ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chạy theo vị trí thanh thép; thường kèm bong tróc, hoen gỉ."
                ),
                "Ảnh (path)": "images/stage2/wall_anmon.png",
            },
        ]
    )

    render_component_crack_table(component_crack_data)

    st.caption(
        "Bảng 2 – Phân loại các vết nứt bê tông thường gặp theo từng loại cấu kiện "
        "(dầm, cột, sàn, tường) – có thể in ra phụ lục kèm hình minh họa."
    )

    st.markdown("### 2.3. Xuất báo cáo kiến thức Stage 2")

    csv_bytes = component_crack_data.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Tải bảng Stage 2 (CSV)",
        data=csv_bytes,
        file_name="BKAI_Stage2_CrackTable.csv",
        mime="text/csv",
        key=f"stage2_csv_{key_prefix}",
    )

    pdf_buf = export_stage2_pdf(component_crack_data)
    st.download_button(
        "📄 Tải báo cáo kiến thức Stage 2 (PDF)",
        data=pdf_buf.getvalue(),
        file_name="BKAI_Stage2_Report.pdf",
        mime="application/pdf",
        key=f"stage2_pdf_{key_prefix}",
    )

# =========================================================
# 5. LƯU THỐNG KÊ NGƯỜI DÙNG
# =========================================================

USER_STATS_FILE = "user_stats.json"

if os.path.exists(USER_STATS_FILE):
    with open(USER_STATS_FILE, "r", encoding="utf-8") as f:
        try:
            user_stats = json.load(f)
        except Exception:
            user_stats = []
else:
    user_stats = []

# =========================================================
# 6. GIAO DIỆN PHÂN TÍCH CHÍNH
# =========================================================

def run_main_app():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)
    with col_title:
        st.title("BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT")
        user = st.session_state.get("username", "")
        if user:
            st.caption(
                f"Xin chào **{user}** – Phân biệt ảnh nứt / không nứt & xuất báo cáo."
            )
        else:
            st.caption("Phân biệt ảnh nứt / không nứt & xuất báo cáo.")

    st.write("---")

    # Form thông tin người dùng
    if "profile_filled" not in st.session_state:
        st.session_state.profile_filled = False

    if not st.session_state.profile_filled:
        st.subheader("Thông tin người sử dụng (bắt buộc trước khi phân tích)")

        with st.form("user_info_form"):
            full_name = st.text_input("Họ và tên *")
            occupation = st.selectbox(
                "Nghề nghiệp / Nhóm đối tượng *",
                [
                    "Sinh viên",
                    "Học viên cao học/ Nghiên cứu sinh",
                    "Kỹ sư xây kết cấu",
                    "Kỹ sư hiện trường (Site Engineer)",
                    "Đơn vị tư vấn giám sát (TVGS)",
                    "Nhà thầu thi công xây dựng",
                    "Chủ đầu tư, Quản Lý Dự án",
                    "Kỹ sư IT",
                    "Khác",
                ],
            )
            email = st.text_input("Email *")

            submit_info = st.form_submit_button("Lưu thông tin & bắt đầu phân tích")

        if submit_info:
            if not full_name or not occupation or not email:
                st.warning("Vui lòng điền đầy đủ Họ tên, Nghề nghiệp và Email.")
            elif "@" not in email or "." not in email:
                st.warning("Email không hợp lệ, vui lòng kiểm tra lại.")
            else:
                st.session_state.profile_filled = True
                st.session_state.user_full_name = full_name
                st.session_state.user_occupation = occupation
                st.session_state.user_email = email

                record = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "login_user": st.session_state.get("username", ""),
                    "full_name": full_name,
                    "occupation": occupation,
                    "email": email,
                }
                user_stats.append(record)
                try:
                    with open(USER_STATS_FILE, "w", encoding="utf-8") as f:
                        json.dump(user_stats, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    st.warning(f"Lưu thống kê người dùng bị lỗi: {e}")

                st.success("Đã lưu thông tin. Bạn có thể tải ảnh lên để phân tích.")

        if not st.session_state.profile_filled:
            return

    # Sidebar
    st.sidebar.header("Cấu hình phân tích")
    min_conf = st.sidebar.slider(
        "Ngưỡng confidence tối thiểu", 0.0, 1.0, 0.3, 0.05
    )
    st.sidebar.caption("Chỉ hiển thị những vết nứt có độ tin cậy ≥ ngưỡng này.")

    with st.sidebar.expander("📊 Quản lý thống kê người dùng"):
        if user_stats:
            df_stats = pd.DataFrame(user_stats)
            st.dataframe(df_stats, use_container_width=True, height=200)
            stats_csv = df_stats.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ Tải thống kê người dùng (CSV)",
                data=stats_csv,
                file_name="BKAI_UserStats.csv",
                mime="text/csv",
            )
        else:
            st.info("Chưa có dữ liệu thống kê người dùng.")

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
                    resp = requests.post(
                        ROBOFLOW_FULL_URL,
                        files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
                        timeout=60,
                    )
                except Exception as e:
                    st.error(f"Lỗi gọi API Roboflow cho ảnh {uploaded_file.name}: {e}")
                    continue

            if resp.status_code != 200:
                st.error(f"Roboflow trả lỗi cho ảnh {uploaded_file.name}.")
                st.text(resp.text[:2000])
                continue

            result = resp.json()
            predictions = result.get("predictions", [])
            preds_conf = [
                p for p in predictions if float(p.get("confidence", 0)) >= min_conf
            ]

            t1 = time.time()
            total_time = t1 - t0

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ảnh gốc")
                st.image(orig_img, use_column_width=True)

            analyzed_img = None
            with col2:
                st.subheader("Ảnh phân tích")
                if len(preds_conf) == 0:
                    # Trường hợp KHÔNG có vết nứt
                    st.image(orig_img, use_column_width=True)
                    st.success("✅ Kết luận: **Không phát hiện vết nứt rõ ràng**.")

                    pdf_no_crack = export_pdf_no_crack(orig_img)
                    st.download_button(
                        "📄 Tải báo cáo PDF (Không có vết nứt)",
                        data=pdf_no_crack.getvalue(),
                        file_name=f"BKAI_NoCrack_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf",
                        key=f"pdf_no_crack_{idx}",
                    )

                    # Không cần Stage 1 & Stage 2 cho ảnh này
                    continue

                else:
                    # Có vết nứt
                    analyzed_img = draw_predictions_with_mask(
                        orig_img, preds_conf, min_conf
                    )
                    st.image(analyzed_img, use_column_width=True)
                    st.error("⚠️ Kết luận: **CÓ vết nứt trên ảnh.**")

            # Nếu tới đây thì CHỈ có trường hợp có vết nứt
            st.write("---")
            tab_stage1, tab_stage2 = st.tabs(
                ["Stage 1 – Báo cáo chi tiết", "Stage 2 – Phân loại vết nứt"]
            )

            # ================== STAGE 1 ==================
            with tab_stage1:
                st.subheader("Bảng thông tin vết nứt")

                confs = [float(p.get("confidence", 0)) for p in preds_conf]
                avg_conf = sum(confs) / len(confs)
                map_val = round(min(1.0, avg_conf - 0.05), 2)

                max_ratio = 0.0
                max_p = preds_conf[0]
                for p in preds_conf:
                    w = float(p.get("width", 0))
                    h = float(p.get("height", 0))
                    ratio = (w * h) / (img_w * img_h)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        max_p = p

                crack_area_ratio = round(max_ratio * 100, 2)
                severity = estimate_severity(max_p, img_w, img_h)

                metrics = [
                    {
                        "vi": "Tên ảnh",
                        "en": "Image Name",
                        "value": uploaded_file.name,
                        "desc": "File ảnh người dùng tải lên",
                    },
                    {
                        "vi": "Thời gian xử lý",
                        "en": "Total Processing Time",
                        "value": f"{total_time:.2f} s",
                        "desc": "Tổng thời gian thực hiện toàn bộ quy trình",
                    },
                    {
                        "vi": "Tốc độ mô hình AI",
                        "en": "Inference Speed",
                        "value": f"{total_time:.2f} s/image",
                        "desc": "Thời gian xử lý mỗi ảnh",
                    },
                    {
                        "vi": "Độ tin cậy (Confidence)",
                        "en": "Confidence",
                        "value": f"{avg_conf:.2f}",
                        "desc": "Mức tin cậy trung bình của mô hình",
                    },
                    {
                        "vi": "mAP (Độ chính xác trung bình)",
                        "en": "Mean Average Precision",
                        "value": f"{map_val:.2f}",
                        "desc": "Độ chính xác định vị vùng nứt (ước lượng từ Confidence).",
                    },
                    {
                        "vi": "Phần trăm vùng nứt",
                        "en": "Crack Area Ratio",
                        "value": f"{crack_area_ratio:.2f} %",
                        "desc": "Diện tích vùng nứt lớn nhất / tổng diện tích ảnh.",
                    },
                    {
                        "vi": "Chiều dài vết nứt",
                        "en": "Crack Length",
                        "value": "—",
                        "desc": "Có thể ước lượng nếu biết tỉ lệ pixel-thực tế.",
                    },
                    {
                        "vi": "Chiều rộng vết nứt",
                        "en": "Crack Width",
                        "value": "—",
                        "desc": "Độ rộng lớn nhất của vết nứt (cần thang đo chuẩn).",
                    },
                    {
                        "vi": "Mức độ nguy hiểm",
                        "en": "Severity Level",
                        "value": severity,
                        "desc": "Phân cấp theo diện tích tương đối vùng nứt lớn nhất.",
                    },
                    {
                        "vi": "Thời gian phân tích",
                        "en": "Timestamp",
                        "value": datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "desc": "Thời điểm thực hiện phân tích.",
                    },
                    {
                        "vi": "Nhận xét tổng quan",
                        "en": "Summary",
                        "value": (
                            "Vết nứt có nguy cơ, cần kiểm tra thêm."
                            if "Nguy hiểm" in severity
                            else "Vết nứt nhỏ, nên tiếp tục theo dõi."
                        ),
                        "desc": "Kết luận tự động của hệ thống.",
                    },
                ]

                metrics_df = pd.DataFrame(metrics)
                styled_df = metrics_df.style.set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#1e88e5"),
                                ("color", "white"),
                                ("font-weight", "bold"),
                            ],
                        },
                        {
                            "selector": "td",
                            "props": [("background-color", "#fafafa")],
                        },
                    ]
                )
                st.dataframe(styled_df, use_container_width=True)

                st.subheader("Biểu đồ thống kê")
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    fig1 = plt.figure(figsize=(4, 3))
                    plt.bar(range(1, len(confs) + 1), confs)
                    plt.xlabel("Crack #")
                    plt.ylabel("Confidence")
                    plt.ylim(0, 1)
                    plt.title("Độ tin cậy từng vùng nứt")
                    st.pyplot(fig1)
                    bar_png = fig_to_png(fig1)
                    plt.close(fig1)

                with col_chart2:
                    labels = ["Vùng nứt lớn nhất", "Phần ảnh còn lại"]
                    sizes = [max_ratio, 1 - max_ratio]
                    fig2 = plt.figure(figsize=(4, 3))
                    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
                    plt.title("Tỷ lệ vùng nứt so với toàn ảnh")
                    st.pyplot(fig2)
                    pie_png = fig_to_png(fig2)
                    plt.close(fig2)

                pdf_buf = export_pdf(
                    original_img=orig_img,
                    analyzed_img=analyzed_img,
                    metrics_df=metrics_df,
                    chart_bar_png=bar_png,
                    chart_pie_png=pie_png,
                )

                st.download_button(
                    "📄 Tải báo cáo PDF cho ảnh này",
                    data=pdf_buf.getvalue(),
                    file_name=f"BKAI_CrackReport_{uploaded_file.name.split('.')[0]}.pdf",
                    mime="application/pdf",
                    key=f"pdf_btn_{idx}_{uploaded_file.name}",
                )

            # ================== STAGE 2 ==================
            with tab_stage2:
                show_stage2_demo(key_prefix=f"stage2_{idx}")

# =========================================================
# 7. ĐĂNG KÝ / ĐĂNG NHẬP
# =========================================================

USERS_FILE = "users.json"

if os.path.exists(USERS_FILE):
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        try:
            users = json.load(f)
        except Exception:
            users = {}
else:
    users = {}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

def show_auth_page():
    # Khung trên cùng: logo + tiêu đề
    col_logo, col_header = st.columns([1, 3])

    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=90)
        else:
            st.markdown("### BKAI")

    with col_header:
        st.title("BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT BÊ TÔNG")
        st.markdown(
            "<p style='font-size:16px;color:#555;'>"
            "Vui lòng đăng nhập hoặc đăng ký để sử dụng hệ thống phân tích vết nứt bê tông."
            "</p>",
            unsafe_allow_html=True,
        )

    st.write("---")

    # Tabs: Đăng nhập / Đăng ký
    tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])

    with tab_login:
        st.subheader("Đăng nhập tài khoản BKAI")
        login_user = st.text_input("Tên đăng nhập", key="login_user")
        login_pass = st.text_input("Mật khẩu", type="password", key="login_pass")

        if st.button("Đăng nhập"):
            if login_user in users and users[login_user] == login_pass:
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.success(f"Đăng nhập thành công! Xin chào, {login_user} 👋")
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu.")

    with tab_register:
        st.subheader("Tạo tài khoản mới")
        reg_user = st.text_input("Tên đăng nhập mới", key="reg_user")
        reg_pass = st.text_input("Mật khẩu mới", type="password", key="reg_pass")
        reg_pass2 = st.text_input("Nhập lại mật khẩu", type="password", key="reg_pass2")

        if st.button("Tạo tài khoản"):
            if not reg_user or not reg_pass:
                st.warning("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")
            elif reg_user in users:
                st.error("Tên đăng nhập đã tồn tại, hãy chọn tên khác.")
            elif reg_pass != reg_pass2:
                st.error("Mật khẩu nhập lại không khớp.")
            else:
                users[reg_user] = reg_pass
                with open(USERS_FILE, "w", encoding="utf-8") as f:
                    json.dump(users, f, ensure_ascii=False, indent=2)
                st.success("Tạo tài khoản thành công! Bạn có thể quay lại tab Đăng nhập.")


# =========================================================
# 8. MAIN ENTRY
# =========================================================

if st.session_state.authenticated:
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.username}")
        if st.button("Đăng xuất"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()
    run_main_app()
else:
    show_auth_page()


