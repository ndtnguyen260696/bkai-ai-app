import streamlit as st
import requests
from PIL import Image, ImageDraw
import colorsys
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
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# =========================================================
# 0. CẤU HÌNH CHUNG
# =========================================================

# A4 xoay ngang cho Stage 2
A4_LANDSCAPE = landscape(A4)

ROBOFLOW_FULL_URL = (
    "https://detect.roboflow.com/crack_segmentation_detection/4"
    "?api_key=nWA6ayjI5bGNpXkkbsAb"
)

LOGO_PATH = "BKAI_Logo.png"

FONT_PATH = "times.ttf"
FONT_NAME = "TimesVN"

# Cấu hình font PDF
if os.path.exists(FONT_PATH):
    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
    except Exception:
        FONT_NAME = "DejaVuSans"
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


def _stable_rgb(i: int):
    """Màu đa dạng theo instance, ổn định theo index. Trả về RGB 0-255."""
    h = (i * 0.17) % 1.0
    s = 0.90
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


@st.cache_resource
def _get_font(size=18):
    """
    FIX lỗi NameError ImageFont trên Streamlit Cloud:
    - Import ImageFont cục bộ trong hàm để không bị shadow/NameError.
    """
    from PIL import ImageFont as PILImageFont

    for fp in [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
        "times.ttf",
    ]:
        try:
            return PILImageFont.truetype(fp, size=size)
        except Exception:
            continue
    return PILImageFont.load_default()


# =========================================================
# 1.1 PARSE POLYGON + TÍNH DIỆN TÍCH MASK
# =========================================================

def extract_poly_points(points_field, img_w: int, img_h: int):
    """
    Chuẩn hoá mọi kiểu 'points' của Roboflow về list[(x_pixel, y_pixel)].
    Roboflow thường trả:
      - list dict: [{"x":..,"y":..}, ...]
      - list pair: [(x,y), ...] (hiếm)
      - dict nhiều segment: {"0":[...], "1":[...]} (tuỳ model)
    Tự nhận biết x/y là normalized (0..1) hay pixel.
    """
    pts = []

    def _append_xy(x, y):
        if x is None or y is None:
            return
        try:
            pts.append((float(x), float(y)))
        except Exception:
            return

    # dict nhiều segment
    if isinstance(points_field, dict):
        for k in sorted(points_field.keys(), key=lambda x: str(x)):
            seg = points_field[k]
            if isinstance(seg, list):
                for p in seg:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        _append_xy(p.get("x"), p.get("y"))
                    elif isinstance(p, (list, tuple)) and len(p) == 2:
                        _append_xy(p[0], p[1])

    # list trực tiếp
    elif isinstance(points_field, list):
        for p in points_field:
            if isinstance(p, dict) and "x" in p and "y" in p:
                _append_xy(p.get("x"), p.get("y"))
            elif isinstance(p, (list, tuple)) and len(p) == 2:
                _append_xy(p[0], p[1])

    if len(pts) < 3:
        return []

    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    max_x = max(xs) if xs else 0
    max_y = max(ys) if ys else 0

    # Nếu <=1.5 coi như normalized (0..1)
    x_norm = (max_x <= 1.5)
    y_norm = (max_y <= 1.5)

    out = []
    for x, y in pts:
        if x_norm:
            x = x * img_w
        if y_norm:
            y = y * img_h
        x = max(0.0, min(float(img_w - 1), float(x)))
        y = max(0.0, min(float(img_h - 1), float(y)))
        out.append((x, y))

    return out


def extract_polygons(points_field, img_w: int, img_h: int):
    """Trả về list polygon; mỗi polygon là list[(x_pixel, y_pixel)]."""
    polys = []
    if isinstance(points_field, dict):
        for k in sorted(points_field.keys(), key=lambda x: str(x)):
            seg = points_field[k]
            poly = extract_poly_points(seg, img_w, img_h)
            if len(poly) >= 3:
                polys.append(poly)
    elif isinstance(points_field, list):
        poly = extract_poly_points(points_field, img_w, img_h)
        if len(poly) >= 3:
            polys.append(poly)
    return polys


def crack_area_from_predictions(predictions, img_w: int, img_h: int):
    """Tính diện tích vùng nứt theo MASK polygon (pixel^2) bằng rasterize."""
    if img_w <= 0 or img_h <= 0:
        return 0.0

    mask = Image.new("L", (img_w, img_h), 0)
    d = ImageDraw.Draw(mask)

    for p in predictions:
        pts_raw = p.get("points", None)
        if pts_raw is None:
            continue
        polys = extract_polygons(pts_raw, img_w, img_h)
        for poly in polys:
            if len(poly) >= 3:
                d.polygon(poly, fill=255)

    # nhanh hơn đếm từng pixel: dùng histogram
    hist = mask.histogram()
    white = sum(hist[1:])  # tất cả >0
    return float(white)


def crack_area_ratio_percent(predictions, img_w: int, img_h: int):
    """Trả về (ratio_percent, area_px2)."""
    img_area = float(img_w * img_h) if img_w > 0 and img_h > 0 else 0.0
    area_px2 = crack_area_from_predictions(predictions, img_w, img_h)
    ratio = (area_px2 / img_area) if img_area > 0 else 0.0
    return (ratio * 100.0, area_px2)


# =========================================================
# 1.2 VẼ MASK + BOX KIỂU DETECTRON2
# =========================================================

def draw_predictions_with_mask(image: Image.Image, predictions, min_conf: float = 0.0):
    """
    Detectron2-style:
    - đa màu theo instance
    - màu CHỮ %, BOX, OVERLAY = giống nhau (cùng RGB)
    - label nền đen
    - có polygon -> overlay mask; không có polygon -> overlay bbox fallback
    """
    base = image.convert("RGB")
    W, H = base.size

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(18)

    for i, p in enumerate(predictions):
        conf = float(p.get("confidence", 0.0))
        if conf < float(min_conf):
            continue

        # Roboflow bbox center x,y + width,height
        x = p.get("x")
        y = p.get("y")
        w = p.get("width")
        h = p.get("height")
        if None in (x, y, w, h):
            continue

        x0 = float(x) - float(w) / 2
        y0 = float(y) - float(h) / 2
        x1 = float(x) + float(w) / 2
        y1 = float(y) + float(h) / 2

        x0 = max(0.0, min(W - 1.0, x0))
        y0 = max(0.0, min(H - 1.0, y0))
        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))

        r, g, b = _stable_rgb(i)

        box_color    = (r, g, b, 255)
        overlay_fill = (r, g, b, 110)
        overlay_edge = (r, g, b, 255)
        text_color   = (r, g, b, 255)

        pts_raw = p.get("points", None)

        # ✅ polygon/mask
        poly = []
        if pts_raw is not None:
            # nếu points là dict nhiều segment -> lấy polygon đầu tiên để vẽ line đẹp
            polys = extract_polygons(pts_raw, W, H)
            if len(polys) > 0:
                # với crack thường 1 polygon
                poly = polys[0]

        if len(poly) >= 3:
            draw.polygon(poly, fill=overlay_fill)
            draw.line(poly + [poly[0]], fill=overlay_edge, width=1)
        else:
            # fallback: overlay nhẹ theo bbox
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 60))

        # bbox
        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)

        # label
        cls = p.get("class", "crack")
        label = f"{cls} {int(round(conf * 100))}%"
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]

        pad = 6
        lx = x0
        ly = y0 - (th + 2 * pad)
        if ly < 0:
            ly = y0 + 2

        draw.rectangle(
            [lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad],
            fill=(0, 0, 0, 200),
        )
        draw.text((lx + pad, ly + pad), label, fill=text_color, font=font)

    result = Image.alpha_composite(base.convert("RGBA"), overlay)
    return result.convert("RGB")


# =========================================================
# 1.3 SEVERITY (DỰA THEO % MASK)
# =========================================================

def estimate_severity_from_ratio(area_ratio_percent: float):
    """Phân cấp theo % diện tích mask so với ảnh (có thể chỉnh ngưỡng)."""
    r = float(area_ratio_percent)
    if r < 0.2:
        return "Nhỏ"
    elif r < 1.0:
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
    - Dùng canvas.
    - Trang 1: logo + tiêu đề + 2 ảnh + banner kết luận + biểu đồ.
    - Trang 2+: bảng metrics.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    page_w, page_h = A4
    LEFT = 20 * mm
    RIGHT = 20 * mm
    TOP = 20 * mm
    BOTTOM = 20 * mm
    CONTENT_W = page_w - LEFT - RIGHT

    TITLE_FONT = FONT_NAME
    TITLE_SIZE = 18
    BODY_FONT = FONT_NAME
    BODY_SIZE = 10
    SMALL_FONT_SIZE = 8

    def draw_header(page_title, subtitle=None, page_no=None):
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

        # Title
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
        footer = f"BKAI – Concrete Crack Inspection | Generated at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
        c.drawString(LEFT, footer_y, footer)
        if page_no is not None:
            c.drawRightString(page_w - RIGHT, footer_y, f"Page {page_no}")

        content_start_y = y_top - max(logo_h, 15 * mm) - 20 * mm
        return content_start_y

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

    # lấy severity + summary
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

    # PAGE 1
    page_no = 1
    content_top_y = draw_header("BÁO CÁO KẾT QUẢ PHÂN TÍCH", page_no=page_no)

    content_top_y -= 5 * mm
    gap_x = 10 * mm
    slot_w = (CONTENT_W - gap_x) / 2.0
    max_img_h = 90 * mm

    c.setFont(BODY_FONT, 11)
    c.setFillColor(colors.black)
    c.drawString(LEFT, content_top_y + 4 * mm, "Ảnh gốc")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Ảnh phân tích")

    left_bottom = draw_pil_image(original_img, LEFT, content_top_y, slot_w, max_img_h)
    right_bottom = draw_pil_image(analyzed_img, LEFT + slot_w + gap_x, content_top_y, slot_w, max_img_h)
    images_bottom_y = min(left_bottom, right_bottom)

    # banner
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

    # charts
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
        c.drawString(LEFT + chart_slot_w + gap_x, pie_bottom - 10, "Tỷ lệ vùng nứt so với toàn ảnh")

    c.showPage()

    # PAGE 2+ metrics table
    page_no += 1
    subtitle = "Bảng tóm tắt các chỉ số vết nứt"
    content_top_y = draw_header("BÁO CÁO KẾT QUẢ PHÂN TÍCH", subtitle=subtitle, page_no=page_no)

    rows = []
    skip_keys = {"Crack Length", "Crack Width"}
    if metrics_df is not None:
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

    header_h = 10 * mm
    base_lead = 4.0
    max_body_y = content_top_y - 10 * mm

    def start_table_page(page_no):
        c.showPage()
        y0 = draw_header("BÁO CÁO KẾT QUẢ PHÂN TÍCH", subtitle=subtitle, page_no=page_no)
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
        draw_wrapped_cell(val, x2, current_y, col3_w, BODY_FONT, BODY_SIZE, leading)

        current_y -= row_h

    c.save()
    buf.seek(0)
    return buf


# =========================================================
# PDF CHO TRƯỜNG HỢP KHÔNG CÓ VẾT NỨT
# =========================================================

def export_pdf_no_crack(original_img):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    page_w, page_h = A4
    LEFT = 20 * mm
    RIGHT = 20 * mm
    TOP = 20 * mm
    BOTTOM = 20 * mm
    CONTENT_W = page_w - LEFT - RIGHT

    TITLE_FONT = FONT_NAME
    BODY_FONT = FONT_NAME

    def draw_header_no_crack():
        y_top = page_h - TOP

        logo_h = 0
        if os.path.exists(LOGO_PATH):
            try:
                logo = ImageReader(LOGO_PATH)
                logo_w = 30 * mm
                iw, ih = logo.getSize()
                logo_h = logo_w * ih / iw
                c.drawImage(logo, LEFT, y_top - logo_h, width=logo_w, height=logo_h, mask="auto")
            except Exception:
                logo_h = 0

        c.setFont(TITLE_FONT, 18)
        c.drawCentredString(page_w / 2, y_top - 6 * mm, "BÁO CÁO KẾT QUẢ PHÂN TÍCH")
        c.setFont(BODY_FONT, 11)
        c.drawCentredString(page_w / 2, y_top - 14 * mm, "Trường hợp: Không phát hiện vết nứt rõ ràng")

        content_top = y_top - max(logo_h, 15 * mm) - 20 * mm
        return content_top

    content_top_y = draw_header_no_crack()

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
    _ = draw_pil(original_img, LEFT + slot_w + gap_x, content_top_y)

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

    footer_y = BOTTOM - 6
    c.setFont(BODY_FONT, 8)
    c.setFillColor(colors.grey)
    c.drawString(LEFT, footer_y, f"BKAI – Concrete Crack Inspection | Generated at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    c.drawRightString(page_w - RIGHT, footer_y, "Page 1")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


# =========================================================
# 3. XUẤT PDF STAGE 2 (KIẾN THỨC, LANDSCAPE)
# =========================================================

def export_stage2_pdf(component_df: pd.DataFrame) -> io.BytesIO:
    left_margin = 20 * mm
    right_margin = 20 * mm
    top_margin = 20 * mm
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

    if os.path.exists(LOGO_PATH):
        logo_flow = RLImage(LOGO_PATH, width=28 * mm, height=28 * mm)
        header_table = Table(
            [[logo_flow, Paragraph("BKAI – BÁO CÁO KIẾN THỨC VẾT NỨT (STAGE 2)", title_style)]],
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
        elements.append(Paragraph("BKAI – BÁO CÁO KIẾN THỨC VẾT NỨT (STAGE 2)", title_style))

    elements.append(
        Paragraph(
            "Bảng phân loại các vết nứt bê tông thường gặp theo từng loại cấu kiện (dầm, cột, sàn, tường).",
            subtitle_style,
        )
    )

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
            0.12 * usable_width,
            0.18 * usable_width,
            0.30 * usable_width,
            0.25 * usable_width,
            0.15 * usable_width,
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
    h3.markdown(f"<div style='{header_style}'>Nguyên nhân hình thành vết nứt</div>", unsafe_allow_html=True)
    h4.markdown(f"<div style='{header_style}'>Đặc trưng về hình dạng vết nứt</div>", unsafe_allow_html=True)
    h5.markdown(f"<div style='{header_style}'>Hình ảnh minh họa vết nứt</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:2px 0 6px 0;'>", unsafe_allow_html=True)

    for component, subdf in component_df.groupby("Cấu kiện"):
        st.markdown(
            f"<div style='background-color:#bbdefb;padding:4px 10px;margin:4px 0;"
            f"font-weight:bold;border-left:4px solid #1976d2;'>"
            f"{str(component).upper()}</div>",
            unsafe_allow_html=True,
        )

        first_row = True
        for _, row in subdf.iterrows():
            c1, c2, c3, c4, c5 = st.columns([1, 1.2, 2.2, 2.2, 1.6])

            if first_row:
                c1.markdown(f"<div style='padding:4px;font-weight:bold;'>{component}</div>", unsafe_allow_html=True)
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

        st.markdown("<hr style='margin:6px 0 10px 0;border-top:1px dashed #b0bec5;'>", unsafe_allow_html=True)


def show_stage2_demo(key_prefix="stage2"):
    st.subheader("Stage 2 – Phân loại vết nứt & gợi ý nguyên nhân / biện pháp")

    st.markdown("### 2.0. Sơ đồ & ví dụ vết nứt trên kết cấu")
    col_img1, col_img2 = st.columns([3, 4])
    with col_img1:
        tree_path = "images/stage2_crack_tree.png"
        if os.path.exists(tree_path):
            st.image(tree_path, caption="Sơ đồ phân loại các loại vết nứt (Stage 2)", use_container_width=True)
        else:
            st.info("Chưa thấy images/stage2_crack_tree.png")

    with col_img2:
        example_path = "images/stage2_structural_example.png"
        if os.path.exists(example_path):
            st.image(example_path, caption="Ví dụ vết nứt trên cấu kiện", use_container_width=True)
        else:
            st.info("Chưa thấy images/stage2_structural_example.png")

    st.markdown("---")

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
    st.selectbox("Chọn loại vết nứt (tóm tắt):", options, key=f"{key_prefix}_summary_selectbox")
    st.caption("Bảng 1 – Tổng hợp dạng nứt theo cơ chế (dùng làm phụ lục).")

    st.subheader("Phân loại các vết nứt bê tông theo cấu kiện")

    component_crack_data = pd.DataFrame(
        [
            {"Cấu kiện":"Dầm","Loại vết nứt":"Vết nứt uốn","Nguyên nhân":"Do mô men uốn vượt giới hạn; cốt thép chịu uốn/tiết diện không đủ.","Đặc trưng hình dạng":"Nhiều ở giữa nhịp; rộng nhất vùng chịu kéo.","Ảnh (path)":"images/stage2/beam_uon.png"},
            {"Cấu kiện":"Dầm","Loại vết nứt":"Vết nứt cắt","Nguyên nhân":"Lực cắt lớn; khả năng chịu cắt bê tông/cốt đai không đủ.","Đặc trưng hình dạng":"Xiên ~45° so với trục dầm.","Ảnh (path)":"images/stage2/beam_cat.png"},
            {"Cấu kiện":"Dầm","Loại vết nứt":"Vết nứt xoắn","Nguyên nhân":"Thiếu cốt thép chịu xoắn; tiết diện không phù hợp.","Đặc trưng hình dạng":"Chéo, ziczac quanh dầm.","Ảnh (path)":"images/stage2/beam_xoan.png"},
            {"Cấu kiện":"Dầm","Loại vết nứt":"Vết nứt ăn mòn cốt thép","Nguyên nhân":"Môi trường xâm thực; lớp bảo vệ mỏng; thép gỉ giãn nở.","Đặc trưng hình dạng":"Chạy dọc theo thép; có thể kèm hoen gỉ/bong lớp bảo vệ.","Ảnh (path)":"images/stage2/beam_anmon.png"},

            {"Cấu kiện":"Cột","Loại vết nứt":"Vết nứt chéo","Nguyên nhân":"Cột chịu nén-uốn/cắt lớn; vật liệu/thiết kế không đủ.","Đặc trưng hình dạng":"Xiên trên bề mặt khi tải gần/vượt khả năng chịu tải.","Ảnh (path)":"images/stage2/column_cheo.png"},
            {"Cấu kiện":"Cột","Loại vết nứt":"Vết nứt tách (dọc)","Nguyên nhân":"Ứng suất nén lớn gây tách dọc; bê tông yếu; cốt dọc không đủ.","Đặc trưng hình dạng":"Nhiều vết dọc song song.","Ảnh (path)":"images/stage2/column_tach.png"},

            {"Cấu kiện":"Sàn","Loại vết nứt":"Vết nứt co ngót dẻo","Nguyên nhân":"Bốc hơi nước nhanh khi bê tông còn dẻo (gió/nóng/khô).","Đặc trưng hình dạng":"Nông, nhỏ; dạng đa giác.","Ảnh (path)":"images/stage2/slab_congot_deo.png"},
            {"Cấu kiện":"Sàn","Loại vết nứt":"Vết nứt co ngót khô","Nguyên nhân":"Co ngót sau đông cứng trong môi trường khô/nóng.","Đặc trưng hình dạng":"Mạng lưới (map cracking) hoặc đường thẳng.","Ảnh (path)":"images/stage2/slab_congot_kho.png"},

            {"Cấu kiện":"Tường bê tông","Loại vết nứt":"Vết nứt co ngót","Nguyên nhân":"Bốc hơi nước nhanh; ứng suất co ngót vượt khả năng chịu kéo.","Đặc trưng hình dạng":"Ngẫu nhiên, đa giác/bắt chéo.","Ảnh (path)":"images/stage2/wall_congot.png"},
            {"Cấu kiện":"Tường bê tông","Loại vết nứt":"Vết nứt do nhiệt","Nguyên nhân":"Chênh lệch nhiệt độ trong bề dày tường.","Đặc trưng hình dạng":"Thường thẳng đứng; rộng hơn vùng chịu kéo do nhiệt.","Ảnh (path)":"images/stage2/wall_nhiet.png"},
        ]
    )

    render_component_crack_table(component_crack_data)
    st.caption("Bảng 2 – Mapping theo cấu kiện (có hình minh hoạ).")

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
    if not ROBOFLOW_FULL_URL:
        st.error("❌ Chưa cấu hình ROBOFLOW_API_KEY.")
        st.stop()

    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)
    with col_title:
        st.title("BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT")
        user = st.session_state.get("username", "")
        if user:
            st.caption(f"Xin chào **{user}** – Phân tích ảnh & xuất báo cáo.")
        else:
            st.caption("Phân tích ảnh & xuất báo cáo.")

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
    min_conf = st.sidebar.slider("Ngưỡng confidence tối thiểu", 0.0, 1.0, 0.3, 0.05)
    st.sidebar.caption("Chỉ hiển thị những vết nứt có độ tin cậy ≥ ngưỡng này.")

    with st.sidebar.expander("📊 Quản lý thống kê người dùng"):
        if user_stats:
            df_stats = pd.DataFrame(user_stats)
            st.dataframe(df_stats, use_container_width=True, height=200)
            stats_csv = df_stats.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇ Tải thống kê người dùng (CSV)", data=stats_csv,
                               file_name="BKAI_UserStats.csv", mime="text/csv")
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

            try:
                result = resp.json()
            except Exception:
                st.error("Không parse được JSON từ Roboflow.")
                st.text(resp.text[:2000])
                continue

            predictions = result.get("predictions", [])
            preds_conf = [p for p in predictions if float(p.get("confidence", 0)) >= float(min_conf)]

            t1 = time.time()
            total_time = t1 - t0

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ảnh gốc")
                st.image(orig_img, use_container_width=True)

            analyzed_img = None
            with col2:
                st.subheader("Ảnh phân tích")
                if len(preds_conf) == 0:
                    st.image(orig_img, use_container_width=True)
                    st.success("✅ Kết luận: **Không phát hiện vết nứt rõ ràng**.")

                    pdf_no_crack = export_pdf_no_crack(orig_img)
                    st.download_button(
                        "📄 Tải báo cáo PDF (Không có vết nứt)",
                        data=pdf_no_crack.getvalue(),
                        file_name=f"BKAI_NoCrack_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf",
                        key=f"pdf_no_crack_{idx}",
                    )
                    continue
                else:
                    analyzed_img = draw_predictions_with_mask(orig_img, preds_conf, min_conf)
                    st.image(analyzed_img, use_container_width=True)
                    st.error("⚠️ Kết luận: **CÓ vết nứt trên ảnh.**")

            # Tabs
            st.write("---")
            tab_stage1, tab_stage2 = st.tabs(["Stage 1 – Báo cáo chi tiết", "Stage 2 – Phân loại vết nứt"])

            # ================== STAGE 1 ==================
            with tab_stage1:
                st.subheader("Bảng thông tin vết nứt")

                confs = [float(p.get("confidence", 0)) for p in preds_conf]
                avg_conf = (sum(confs) / len(confs)) if confs else 0.0

                crack_ratio_percent, crack_area_px2 = crack_area_ratio_percent(preds_conf, img_w, img_h)
                severity = estimate_severity_from_ratio(crack_ratio_percent)

                metrics = [
                    {"vi": "Tên ảnh", "en": "Image Name", "value": uploaded_file.name, "desc": "File ảnh người dùng tải lên"},
                    {"vi": "Thời gian xử lý", "en": "Total Processing Time", "value": f"{total_time:.2f} s", "desc": "Tổng thời gian xử lý"},
                    {"vi": "Tốc độ mô hình AI", "en": "Inference Speed", "value": f"{total_time:.2f} s/image", "desc": "Thời gian/ảnh"},
                    {"vi": "Độ tin cậy trung bình", "en": "Average Confidence", "value": f"{avg_conf:.2f}", "desc": "Trung bình confidence"},
                    {"vi": "Diện tích vùng nứt", "en": "Crack Area (px^2)", "value": f"{crack_area_px2:.0f}", "desc": "Diện tích mask theo pixel"},
                    {"vi": "Phần trăm vùng nứt", "en": "Crack Area Ratio", "value": f"{crack_ratio_percent:.2f} %", "desc": "Mask/diện tích ảnh (%)"},
                    {"vi": "Chiều dài vết nứt", "en": "Crack Length", "value": "—", "desc": "Ước lượng nếu có tỉ lệ pixel-thực tế"},
                    {"vi": "Chiều rộng vết nứt", "en": "Crack Width", "value": "—", "desc": "Cần thang đo chuẩn"},
                    {"vi": "Mức độ nguy hiểm", "en": "Severity Level", "value": severity, "desc": "Phân cấp theo % mask"},
                    {"vi": "Thời gian phân tích", "en": "Timestamp", "value": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "desc": "Thời điểm chạy"},
                    {
                        "vi": "Nhận xét tổng quan",
                        "en": "Summary",
                        "value": ("Vết nứt có nguy cơ, cần kiểm tra thêm." if "Nguy hiểm" in severity else "Vết nứt nhỏ/trung bình, nên tiếp tục theo dõi."),
                        "desc": "Kết luận tự động",
                    },
                ]

                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("Biểu đồ thống kê")
                col_chart1, col_chart2 = st.columns(2)

                bar_png = None
                pie_png = None

                # ---------- BAR CHART (đẹp + cột mảnh, 1 crack không bị to) ----------
                with col_chart1:
                    fig1, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)

                    xs = list(range(1, len(confs) + 1))
                    ax.bar(xs, confs, width=0.35)  # cột mảnh

                    ax.set_ylim(0, 1)
                    ax.set_xticks(xs)
                    ax.set_xlabel("Crack #")
                    ax.set_ylabel("Confidence")
                    ax.set_title("Độ tin cậy từng vùng nứt", pad=8)

                    ax.grid(axis="y", alpha=0.25)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    for x, v in zip(xs, confs):
                        ax.text(x, v + 0.02, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8)

                    if len(xs) == 1:
                        ax.set_xlim(0.5, 1.5)

                    fig1.tight_layout()
                    st.pyplot(fig1)

                    bar_png = fig_to_png(fig1)
                    plt.close(fig1)

                # ---------- PIE CHART ----------
                with col_chart2:
                    labels = ["Vùng nứt (mask)", "Phần ảnh còn lại"]
                    ratio = crack_ratio_percent / 100.0
                    ratio = max(0.0, min(1.0, ratio))
                    sizes = [ratio, 1 - ratio]

                    fig2, ax2 = plt.subplots(figsize=(4.2, 3.2), dpi=150)
                    ax2.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
                    ax2.set_title("Tỷ lệ vùng nứt so với toàn ảnh", pad=8)
                    fig2.tight_layout()

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
# 7. ĐĂNG KÝ / ĐĂNG NHẬP (GIỮ NGUYÊN KIỂU JSON NHƯ BẠN)
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
    col_logo, col_header = st.columns([1, 3])

    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=90)
        else:
            st.markdown("### BKAI")

    with col_header:
        st.markdown(
            "<h2 style='margin:5px 0 5px 0; color:#333;'>"
            "BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT BÊ TÔNG"
            "</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:15px; color:#555;'>"
            "Vui lòng đăng nhập hoặc đăng ký để sử dụng hệ thống."
            "</p>",
            unsafe_allow_html=True,
        )

    st.write("---")

    tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])

    with tab_login:
        st.subheader("Đăng nhập tài khoản BKAI")
        login_user = st.text_input("Tên đăng nhập", key="login_user")
        login_pass = st.text_input("Mật khẩu", type="password", key="login_pass")

        if st.button("Đăng nhập"):
            if login_user in users and users.get(login_user) == login_pass:
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
