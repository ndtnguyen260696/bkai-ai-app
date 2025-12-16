# =========================================================
# BKAI – Streamlit Cloud Safe App (Roboflow + PDF Report)
# 1 FILE DUY NHẤT: app.py
# =========================================================

import os
import io
import json
import time
import datetime
from pathlib import Path

import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageDraw

# Matplotlib safe for server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================================================
# 0) STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT",
    layout="wide",
)

A4_LANDSCAPE = landscape(A4)

# Root paths (Cloud safe)
APP_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
DATA_DIR = APP_DIR / "data"
IMG_DIR = APP_DIR / "images"
STAGE2_DIR = IMG_DIR / "stage2"

DATA_DIR.mkdir(parents=True, exist_ok=True)  # Cloud: OK, but may reset when redeploy

# Assets
LOGO_PATH = str(APP_DIR / "BKAI_Logo.png")     # đặt logo tại root repo
FONT_PATH = str(APP_DIR / "times.ttf")         # nếu có

# JSON files (best effort; Cloud can write but not permanent long-term)
USERS_FILE = DATA_DIR / "users.json"
USER_STATS_FILE = DATA_DIR / "user_stats.json"


# =========================================================
# 1) SAFE ROBOTFLOW CONFIG (NO HARD-CODE KEY)
# =========================================================
def get_secret(name: str, default: str = "") -> str:
    # ưu tiên Streamlit Secrets, fallback env
    try:
        val = st.secrets.get(name, None)
        if val is not None:
            return str(val).strip()
    except Exception:
        pass
    return str(os.getenv(name, default)).strip()

ROBOFLOW_API_KEY = get_secret("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL   = get_secret("ROBOFLOW_MODEL")
ROBOFLOW_VERSION = get_secret("ROBOFLOW_VERSION")

def roboflow_is_configured() -> bool:
    return bool(ROBOFLOW_API_KEY and ROBOFLOW_MODEL and ROBOFLOW_VERSION)

def build_roboflow_url() -> str:
    return f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"


# =========================================================
# 2) FONTS (CLOUD SAFE)
# =========================================================
# ƯU TIÊN: times.ttf nếu bạn có, nếu không thì dùng DejaVuSans (gần như chắc có trên Linux)
FONT_NAME = "DejaVuSans"

def register_fonts():
    global FONT_NAME

    # Try times.ttf
    if os.path.exists(FONT_PATH):
        try:
            pdfmetrics.registerFont(TTFont("TimesVN", FONT_PATH))
            FONT_NAME = "TimesVN"
            return
        except Exception:
            pass

    # Fallback DejaVuSans
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("DejaVuSans", p))
                FONT_NAME = "DejaVuSans"
                return
            except Exception:
                pass

register_fonts()


# =========================================================
# 3) JSON SAFE IO (CLOUD SAFE)
# =========================================================
def safe_read_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def safe_write_json(path: Path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# Load users/stats into session (avoid crash if file can’t write)
if "users" not in st.session_state:
    st.session_state.users = safe_read_json(USERS_FILE, default={})

if "user_stats" not in st.session_state:
    st.session_state.user_stats = safe_read_json(USER_STATS_FILE, default=[])


# =========================================================
# 4) COMMON UTILS
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
    green_fill  = (0, 255, 0, 80)

    for p in predictions:
        conf = float(p.get("confidence", 0))
        if conf < min_conf:
            continue

        x = p.get("x"); y = p.get("y")
        w = p.get("width"); h = p.get("height")
        if None in (x, y, w, h):
            continue

        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        draw.rectangle([x0, y0, x1, y1], outline=green_solid, width=3)
        cls = p.get("class", "crack")
        label = f"{cls} {conf:.2f}"
        draw.text((x0 + 3, y0 + 3), label, fill=green_solid)

        # Polygon points (safe)
        flat_pts = []
        try:
            pts_raw = p.get("points")
            if pts_raw is not None:
                flat_pts = extract_poly_points(pts_raw)
        except Exception:
            flat_pts = []

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
# 5) ROBOTFLOW CALL (ROBUST)
# =========================================================
def call_roboflow(image_bytes: bytes, filename="image.jpg", timeout=60):
    """
    Returns: (ok: bool, payload: dict|str, status_code: int)
    """
    if not roboflow_is_configured():
        return False, {
            "error": "Missing Roboflow config",
            "hint": "Set ROBOFLOW_API_KEY / ROBOFLOW_MODEL / ROBOFLOW_VERSION in Streamlit Secrets."
        }, 0

    url = build_roboflow_url()
    headers = {"User-Agent": "BKAI-Streamlit/1.0"}

    try:
        resp = requests.post(
            url,
            files={"file": (filename, image_bytes, "image/jpeg")},
            headers=headers,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return False, {"error": "Timeout", "hint": "Request timed out. Try again."}, 408
    except Exception as e:
        return False, {"error": "Request failed", "detail": str(e)}, 0

    status = resp.status_code

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    if status == 200:
        if isinstance(data, dict):
            return True, data, status
        return False, {"error": "Invalid JSON response", "raw": str(data)[:2000]}, status

    if status in (401, 403):
        return False, {
            "error": "Forbidden / Unauthorized",
            "status_code": status,
            "raw": data,
            "fix": [
                "1) API key đúng chưa? (tạo key mới nếu cần)",
                "2) Model/Version đúng chưa? (tên model, số version)",
                "3) Project Roboflow có Private không? Key có quyền Hosted Inference không?",
                "4) Key có bị revoke do lộ trên GitHub không?"
            ]
        }, status

    return False, {"error": "Roboflow error", "status_code": status, "raw": data}, status


# =========================================================
# 6) PDF EXPORTS
# =========================================================
def export_pdf(original_img, analyzed_img, metrics_df, chart_bar_png=None, chart_pie_png=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    page_w, page_h = A4
    LEFT = 20 * mm
    RIGHT = 20 * mm
    TOP = 20 * mm
    BOTTOM = 20 * mm
    CONTENT_W = page_w - LEFT - RIGHT

    TITLE_SIZE = 18
    BODY_SIZE = 10
    SMALL_SIZE = 8

    def draw_header(page_title, subtitle=None, page_no=None):
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

        c.setFillColor(colors.black)
        c.setFont(FONT_NAME, TITLE_SIZE)
        c.drawCentredString(page_w / 2.0, y_top - 6 * mm, page_title)

        if subtitle:
            c.setFont(FONT_NAME, 11)
            c.drawCentredString(page_w / 2.0, y_top - 13 * mm, subtitle)

        footer_y = BOTTOM - 6
        c.setFont(FONT_NAME, SMALL_SIZE)
        c.setFillColor(colors.grey)
        footer = f"BKAI – Concrete Crack Inspection | Generated at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
        c.drawString(LEFT, footer_y, footer)
        if page_no is not None:
            c.drawRightString(page_w - RIGHT, footer_y, f"Page {page_no}")

        return y_top - max(logo_h, 15 * mm) - 20 * mm

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

    severity_val = ""
    summary_val = ""
    if metrics_df is not None and len(metrics_df) > 0:
        try:
            for _, row in metrics_df.iterrows():
                en = str(row.get("en", "")).strip().lower()
                if en == "severity level":
                    severity_val = str(row.get("value", ""))
                if en == "summary":
                    summary_val = str(row.get("value", ""))
        except Exception:
            pass

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

    c.setFont(FONT_NAME, 11)
    c.setFillColor(colors.black)
    c.drawString(LEFT, content_top_y + 4 * mm, "Ảnh gốc")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Ảnh phân tích")

    left_bottom = draw_pil_image(original_img, LEFT, content_top_y, slot_w, max_img_h)
    right_bottom = draw_pil_image(analyzed_img, LEFT + slot_w + gap_x, content_top_y, slot_w, max_img_h)
    images_bottom_y = min(left_bottom, right_bottom)

    banner_h = 16 * mm
    banner_bottom = images_bottom_y - 12 * mm
    if banner_bottom < BOTTOM + 40 * mm:
        banner_bottom = BOTTOM + 40 * mm

    c.setFillColor(banner_fill)
    c.rect(LEFT, banner_bottom, CONTENT_W, banner_h, stroke=0, fill=1)
    c.setFillColor(banner_text)
    c.setFont(FONT_NAME, 11)
    c.drawString(LEFT + 4 * mm, banner_bottom + banner_h / 2.0 - 4, summary_val)

    charts_top_y = banner_bottom - 18 * mm
    max_chart_h = 70 * mm

    if chart_bar_png is not None:
        try:
            chart_bar_png.seek(0)
            bar_img = ImageReader(chart_bar_png)
            bw, bh = bar_img.getSize()
            scale_bar = min(slot_w / bw, max_chart_h / bh)
            cw = bw * scale_bar
            ch = bh * scale_bar
            bar_bottom = charts_top_y - ch
            c.drawImage(bar_img, LEFT, bar_bottom, width=cw, height=ch, mask="auto")
            c.setFont(FONT_NAME, 10)
            c.setFillColor(colors.black)
            c.drawString(LEFT, bar_bottom - 10, "Độ tin cậy từng vùng nứt")
        except Exception:
            pass

    if chart_pie_png is not None:
        try:
            chart_pie_png.seek(0)
            pie_img = ImageReader(chart_pie_png)
            pw, ph = pie_img.getSize()
            scale_pie = min(slot_w / pw, max_chart_h / ph)
            cw = pw * scale_pie
            ch = ph * scale_pie
            pie_bottom = charts_top_y - ch
            c.drawImage(pie_img, LEFT + slot_w + gap_x, pie_bottom, width=cw, height=ch, mask="auto")
            c.setFont(FONT_NAME, 10)
            c.setFillColor(colors.black)
            c.drawString(LEFT + slot_w + gap_x, pie_bottom - 10, "Tỷ lệ vùng nứt so với toàn ảnh")
        except Exception:
            pass

    c.showPage()

    # PAGE 2 – table
    page_no += 1
    subtitle = "Bảng tóm tắt các chỉ số vết nứt"
    content_top_y = draw_header("BÁO CÁO KẾT QUẢ PHÂN TÍCH", subtitle=subtitle, page_no=page_no)

    rows = []
    skip_keys = {"Crack Length", "Crack Width"}
    if metrics_df is not None and len(metrics_df) > 0:
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

    def wrap_text(text, font_name, font_size, max_width):
        words = str(text).split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for w in words[1:]:
            trial = current + " " + w
            if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = w
        lines.append(current)
        return lines

    def draw_table_header(top_y, x0, x1, x2):
        c.setFillColor(colors.HexColor("#1e88e5"))
        c.rect(x0, top_y - header_h, CONTENT_W, header_h, stroke=0, fill=1)
        c.setFont(FONT_NAME, 10)
        c.setFillColor(colors.white)
        c.drawString(x0 + 2, top_y - header_h + 3, "No.")
        c.drawString(x1 + 2, top_y - header_h + 3, "Chỉ số (VI / EN)")
        c.drawString(x2 + 2, top_y - header_h + 3, "Giá trị / Value")
        return top_y - header_h

    x0 = LEFT
    x1 = x0 + col1_w
    x2 = x1 + col2_w

    current_y = draw_table_header(content_top_y - 10 * mm, x0, x1, x2)
    leading = BODY_SIZE + 4.0

    for i, (label, val) in enumerate(rows, start=1):
        label_lines = wrap_text(label, FONT_NAME, BODY_SIZE, col2_w - 4)
        value_lines = wrap_text(val, FONT_NAME, BODY_SIZE, col3_w - 4)
        n_lines = max(len(label_lines), len(value_lines))
        row_h = n_lines * leading + 6

        if current_y - row_h < (20 * mm):
            page_no += 1
            c.showPage()
            content_top_y = draw_header("BÁO CÁO KẾT QUẢ PHÂN TÍCH", subtitle=subtitle, page_no=page_no)
            current_y = draw_table_header(content_top_y - 10 * mm, x0, x1, x2)

        if i % 2 == 0:
            c.setFillColor(colors.HexColor("#e3f2fd"))
            c.rect(x0, current_y - row_h, CONTENT_W, row_h, stroke=0, fill=1)

        c.setStrokeColor(colors.grey)
        c.setLineWidth(0.3)
        c.rect(x0, current_y - row_h, CONTENT_W, row_h, stroke=1, fill=0)

        c.setFont(FONT_NAME, BODY_SIZE)
        c.setFillColor(colors.black)
        c.drawString(x0 + 2, current_y - leading, str(i))

        y_text = current_y - leading
        for line in label_lines:
            c.drawString(x1 + 2, y_text, line)
            y_text -= leading

        y_text = current_y - leading
        for line in value_lines:
            c.drawString(x2 + 2, y_text, line)
            y_text -= leading

        current_y -= row_h

    c.save()
    buf.seek(0)
    return buf

def export_pdf_no_crack(original_img):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    page_w, page_h = A4
    LEFT = 20 * mm
    RIGHT = 20 * mm
    TOP = 20 * mm
    BOTTOM = 20 * mm
    CONTENT_W = page_w - LEFT - RIGHT

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

        c.setFont(FONT_NAME, 18)
        c.drawCentredString(page_w / 2, y_top - 6 * mm, "BÁO CÁO KẾT QUẢ PHÂN TÍCH")
        c.setFont(FONT_NAME, 11)
        c.drawCentredString(page_w / 2, y_top - 14 * mm, "Trường hợp: Không phát hiện vết nứt rõ ràng")
        return y_top - max(logo_h, 15 * mm) - 20 * mm

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

    c.setFont(FONT_NAME, 11)
    c.drawString(LEFT, content_top_y + 4 * mm, "Ảnh gốc")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Ảnh phân tích")

    left_bottom = draw_pil(original_img, LEFT, content_top_y)
    _ = draw_pil(original_img, LEFT + slot_w + gap_x, content_top_y)

    banner_y = left_bottom - 12 * mm
    banner_h = 16 * mm

    c.setFillColor(colors.HexColor("#e8f5e9"))
    c.rect(LEFT, banner_y, CONTENT_W, banner_h, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#2e7d32"))
    c.setFont(FONT_NAME, 11)
    c.drawString(
        LEFT + 4 * mm,
        banner_y + banner_h / 2 - 4,
        "Không phát hiện vết nứt rõ ràng trong ảnh theo ngưỡng của mô hình."
    )

    footer_y = BOTTOM - 6
    c.setFont(FONT_NAME, 8)
    c.setFillColor(colors.grey)
    c.drawString(LEFT, footer_y, f"BKAI – Concrete Crack Inspection | Generated at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    c.drawRightString(page_w - RIGHT, footer_y, "Page 1")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


# =========================================================
# 7) STAGE 2 PDF + UI
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

    page_w, _ = A4_LANDSCAPE
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

    header_row = []
    if os.path.exists(LOGO_PATH):
        logo_flow = RLImage(LOGO_PATH, width=28 * mm, height=28 * mm)
        header_row.append(logo_flow)
        header_row.append(Paragraph("BKAI – BÁO CÁO KIẾN THỨC VẾT NỨT (STAGE 2)", title_style))
        header_table = Table([header_row], colWidths=[30 * mm, doc.width - 30 * mm], hAlign="LEFT")
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("GRID", (0, 0), (-1, -1), 0, colors.white),
        ]))
        elements.append(header_table)
    else:
        elements.append(Paragraph("BKAI – BÁO CÁO KIẾN THỨC VẾT NỨT (STAGE 2)", title_style))

    elements.append(Paragraph(
        "Bảng phân loại các vết nứt bê tông thường gặp theo từng loại cấu kiện (dầm, cột, sàn, tường).",
        subtitle_style,
    ))

    data = [[
        Paragraph("Cấu kiện", normal),
        Paragraph("Loại vết nứt", normal),
        Paragraph("Nguyên nhân hình thành vết nứt", normal),
        Paragraph("Đặc trưng về hình dạng vết nứt", normal),
        Paragraph("Hình ảnh minh họa vết nứt", normal),
    ]]

    def make_thumb(path: str):
        if isinstance(path, str) and path and os.path.exists(path):
            return RLImage(path, width=25 * mm, height=25 * mm)
        return Paragraph("—", normal)

    for _, row in component_df.iterrows():
        img_path = row.get("Ảnh (path)", "") or row.get("Hình ảnh minh họa", "")
        data.append([
            Paragraph(str(row["Cấu kiện"]), normal),
            Paragraph(str(row["Loại vết nứt"]), normal),
            Paragraph(str(row["Nguyên nhân"]), normal),
            Paragraph(str(row["Đặc trưng hình dạng"]), normal),
            make_thumb(str(APP_DIR / img_path) if isinstance(img_path, str) and img_path.startswith("images/") else str(img_path)),
        ])

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

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e88e5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-2, -1), 8),
        ("VALIGN", (0, 1), (-1, -1), "TOP"),
        ("ALIGN", (0, 1), (-2, -1), "LEFT"),
        ("ALIGN", (-1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))

    elements.append(table)
    doc.build(elements)
    buf.seek(0)
    return buf

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
            f"{component.upper()}</div>",
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
            # Convert relative images/... to absolute
            abs_path = None
            if isinstance(img_path, str) and img_path:
                abs_path = (APP_DIR / img_path) if img_path.startswith("images/") else Path(img_path)

            if abs_path and abs_path.exists():
                c5.image(str(abs_path), use_container_width=True)
            else:
                c5.write("—")

        st.markdown("<hr style='margin:6px 0 10px 0;border-top:1px dashed #b0bec5;'>", unsafe_allow_html=True)

def show_stage2_demo(key_prefix="stage2"):
    st.subheader("Stage 2 – Phân loại vết nứt & gợi ý nguyên nhân / biện pháp")

    st.markdown("### 2.0. Sơ đồ & ví dụ vết nứt trên kết cấu")
    col_img1, col_img2 = st.columns([3, 4])
    with col_img1:
        tree_path = IMG_DIR / "stage2_crack_tree.png"
        if tree_path.exists():
            st.image(str(tree_path), caption="Sơ đồ phân loại các loại vết nứt theo thời điểm xuất hiện và mức độ ảnh hưởng",
                     use_container_width=True)
        else:
            st.info("Chưa thấy images/stage2_crack_tree.png")
    with col_img2:
        example_path = IMG_DIR / "stage2_structural_example.png"
        if example_path.exists():
            st.image(str(example_path), caption="Ví dụ các loại vết nứt kết cấu bê tông (dầm, cột, tường, sàn)",
                     use_container_width=True)
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
    st.caption("Bảng 1 – Tổng hợp các dạng nứt theo cơ chế hình thành và biện pháp kiểm soát (có thể dùng làm phụ lục).")

    st.subheader("Phân loại các vết nứt bê tông thường xảy ra cho từng loại cấu kiện")

    component_crack_data = pd.DataFrame([
        {"Cấu kiện":"Dầm","Loại vết nứt":"Vết nứt uốn","Nguyên nhân":"Do mô men uốn vượt quá giới hạn chịu tải; thép chịu uốn không đủ.","Đặc trưng hình dạng":"Nứt ở giữa nhịp; rộng nhất vùng chịu kéo.","Ảnh (path)":"images/stage2/beam_uon.png"},
        {"Cấu kiện":"Dầm","Loại vết nứt":"Vết nứt cắt","Nguyên nhân":"Lực cắt lớn; cốt đai không đủ.","Đặc trưng hình dạng":"Nứt xiên ~45°.","Ảnh (path)":"images/stage2/beam_cat.png"},
        {"Cấu kiện":"Cột","Loại vết nứt":"Vết nứt chéo","Nguyên nhân":"Nén-uốn/cắt lớn; vật liệu hoặc cấu tạo không đủ.","Đặc trưng hình dạng":"Nứt xiên trên bề mặt cột.","Ảnh (path)":"images/stage2/column_cheo.png"},
        {"Cấu kiện":"Sàn","Loại vết nứt":"Vết nứt co ngót khô","Nguyên nhân":"Co ngót do bay hơi nước sau đông cứng.","Đặc trưng hình dạng":"Mạng lưới/map cracking.","Ảnh (path)":"images/stage2/slab_congot_kho.png"},
        {"Cấu kiện":"Tường bê tông","Loại vết nứt":"Vết nứt do nhiệt","Nguyên nhân":"Chênh lệch nhiệt độ; co/giãn không đều.","Đặc trưng hình dạng":"Thường thẳng đứng; rộng hơn vùng chịu kéo.","Ảnh (path)":"images/stage2/wall_nhiet.png"},
    ])

    render_component_crack_table(component_crack_data)
    st.caption("Bảng 2 – Phân loại vết nứt theo cấu kiện (dầm, cột, sàn, tường) – có thể in phụ lục kèm hình.")

    st.markdown("### 2.3. Xuất báo cáo kiến thức Stage 2")
    csv_bytes = component_crack_data.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Tải bảng Stage 2 (CSV)", data=csv_bytes, file_name="BKAI_Stage2_CrackTable.csv",
                       mime="text/csv", key=f"stage2_csv_{key_prefix}")

    pdf_buf = export_stage2_pdf(component_crack_data)
    st.download_button("📄 Tải báo cáo kiến thức Stage 2 (PDF)", data=pdf_buf.getvalue(),
                       file_name="BKAI_Stage2_Report.pdf", mime="application/pdf", key=f"stage2_pdf_{key_prefix}")


# =========================================================
# 8) AUTH (LOGIN/REGISTER) – CLOUD SAFE
# =========================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

def show_auth_page():
    users = st.session_state.users

    col_logo, col_header = st.columns([1, 3])
    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=90)
        else:
            st.markdown("### BKAI")
    with col_header:
        st.markdown(
            "<h2 style='margin:5px 0 5px 0; color:#333;'>BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT BÊ TÔNG</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:15px; color:#555;'>Vui lòng đăng nhập hoặc đăng ký để sử dụng hệ thống.</p>",
            unsafe_allow_html=True,
        )

    st.write("---")
    tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])

    with tab_login:
        st.subheader("Đăng nhập tài khoản BKAI")
        login_user = st.text_input("Tên đăng nhập", key="login_user")
        login_pass = st.text_input("Mật khẩu", type="password", key="login_pass")
        if st.button("Đăng nhập", key="btn_login"):
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
        if st.button("Tạo tài khoản", key="btn_register"):
            if not reg_user or not reg_pass:
                st.warning("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")
            elif reg_user in users:
                st.error("Tên đăng nhập đã tồn tại, hãy chọn tên khác.")
            elif reg_pass != reg_pass2:
                st.error("Mật khẩu nhập lại không khớp.")
            else:
                users[reg_user] = reg_pass
                st.session_state.users = users

                # Best-effort save
                saved = safe_write_json(USERS_FILE, users)
                if saved:
                    st.success("Tạo tài khoản thành công! (đã lưu) Bạn có thể quay lại tab Đăng nhập.")
                else:
                    st.warning("Tạo tài khoản thành công! (chạy Cloud có thể không lưu vĩnh viễn) Bạn có thể đăng nhập ngay trong phiên này.")


# =========================================================
# 9) MAIN APP
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
            st.caption(f"Xin chào **{user}** – Phân biệt ảnh nứt / không nứt & xuất báo cáo.")
        else:
            st.caption("Phân biệt ảnh nứt / không nứt & xuất báo cáo.")
    st.write("---")

    # ================= ROBOTFLOW STATUS + TEST =================
    with st.sidebar:
        st.header("Roboflow Status")
        if roboflow_is_configured():
            st.success("Roboflow config OK (key/model/version đã set).")
            st.caption(f"Model: {ROBOFLOW_MODEL} | Version: {ROBOFLOW_VERSION}")
        else:
            st.error("Thiếu Roboflow Secrets (API_KEY / MODEL / VERSION).")

        if st.button("🧪 Test Roboflow API", key="btn_test_rf"):
            test_img = Image.new("RGB", (256, 256), (255, 255, 255))
            bio = io.BytesIO()
            test_img.save(bio, format="JPEG", quality=95)
            ok, payload, status = call_roboflow(bio.getvalue(), filename="test.jpg", timeout=30)
            if ok:
                st.success("Test OK: Roboflow trả về kết quả.")
                st.json(payload)
            else:
                st.error(f"Test FAIL: HTTP {status}")
                st.json(payload)

    # ================= FORM INFO USER =================
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
                return
            if "@" not in email or "." not in email:
                st.warning("Email không hợp lệ, vui lòng kiểm tra lại.")
                return

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
            st.session_state.user_stats.append(record)

            # Best-effort save (Cloud may not persist after redeploy)
            safe_write_json(USER_STATS_FILE, st.session_state.user_stats)

            st.success("Đã lưu thông tin. Bạn có thể tải ảnh lên để phân tích.")
            st.rerun()
        else:
            return

    # ================= SIDEBAR: SETTINGS + STATS =================
    st.sidebar.header("Cấu hình phân tích")
    min_conf = st.sidebar.slider("Ngưỡng confidence tối thiểu", 0.0, 1.0, 0.3, 0.05)
    st.sidebar.caption("Chỉ hiển thị những vết nứt có độ tin cậy ≥ ngưỡng này.")

    with st.sidebar.expander("📊 Quản lý thống kê người dùng"):
        stats = st.session_state.user_stats
        if stats:
            df_stats = pd.DataFrame(stats)
            st.dataframe(df_stats, use_container_width=True, height=200)
            stats_csv = df_stats.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇ Tải thống kê người dùng (CSV)", data=stats_csv,
                               file_name="BKAI_UserStats.csv", mime="text/csv")
        else:
            st.info("Chưa có dữ liệu thống kê người dùng.")

    # ================= UPLOAD + ANALYZE =================
    uploaded_files = st.file_uploader(
        "Tải một hoặc nhiều ảnh bê tông (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    analyze_btn = st.button("🔍 Phân tích ảnh", key="btn_analyze")

    if analyze_btn:
        if not uploaded_files:
            st.warning("Vui lòng chọn ít nhất một ảnh trước khi bấm **Phân tích**.")
            st.stop()

        if not roboflow_is_configured():
            st.error("Thiếu cấu hình Roboflow. Vui lòng set Secrets trước khi phân tích.")
            st.stop()

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            st.write("---")
            st.markdown(f"## Ảnh {idx}: `{uploaded_file.name}`")

            t0 = time.time()
            try:
                orig_img = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Không mở được ảnh {uploaded_file.name}: {e}")
                continue

            img_w, img_h = orig_img.size

            buf = io.BytesIO()
            orig_img.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()

            with st.spinner(f"Đang gửi ảnh {idx} tới mô hình AI trên Roboflow..."):
                ok, payload, status = call_roboflow(img_bytes, filename=uploaded_file.name, timeout=60)

            if not ok:
                st.error(f"Roboflow trả lỗi cho ảnh {uploaded_file.name} (HTTP {status}).")
                st.json(payload)
                continue

            predictions = payload.get("predictions", []) if isinstance(payload, dict) else []
            preds_conf = [p for p in predictions if float(p.get("confidence", 0)) >= min_conf]

            total_time = time.time() - t0

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
                        file_name=f"BKAI_NoCrack_{Path(uploaded_file.name).stem}.pdf",
                        mime="application/pdf",
                        key=f"pdf_no_crack_{idx}",
                    )
                    continue
                else:
                    analyzed_img = draw_predictions_with_mask(orig_img, preds_conf, min_conf)
                    st.image(analyzed_img, use_container_width=True)
                    st.error("⚠️ Kết luận: **CÓ vết nứt trên ảnh.**")

            st.write("---")
            tab_stage1, tab_stage2 = st.tabs(["Stage 1 – Báo cáo chi tiết", "Stage 2 – Phân loại vết nứt"])

            with tab_stage1:
                st.subheader("Bảng thông tin vết nứt")

                confs = [float(p.get("confidence", 0)) for p in preds_conf]
                avg_conf = sum(confs) / max(len(confs), 1)
                map_val = round(min(1.0, max(0.0, avg_conf - 0.05)), 2)

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
                    {"vi": "Tên ảnh", "en": "Image Name", "value": uploaded_file.name, "desc": "File ảnh người dùng tải lên"},
                    {"vi": "Thời gian xử lý", "en": "Total Processing Time", "value": f"{total_time:.2f} s", "desc": "Tổng thời gian thực hiện toàn bộ quy trình"},
                    {"vi": "Tốc độ mô hình AI", "en": "Inference Speed", "value": f"{total_time:.2f} s/image", "desc": "Thời gian xử lý mỗi ảnh"},
                    {"vi": "Độ tin cậy (Confidence)", "en": "Confidence", "value": f"{avg_conf:.2f}", "desc": "Mức tin cậy trung bình của mô hình"},
                    {"vi": "mAP (ước lượng)", "en": "Mean Average Precision", "value": f"{map_val:.2f}", "desc": "Ước lượng từ confidence (mang tính tham khảo)."},
                    {"vi": "Phần trăm vùng nứt", "en": "Crack Area Ratio", "value": f"{crack_area_ratio:.2f} %", "desc": "Diện tích vùng nứt lớn nhất / tổng ảnh."},
                    {"vi": "Mức độ nguy hiểm", "en": "Severity Level", "value": severity, "desc": "Phân cấp theo diện tích tương đối vùng nứt lớn nhất."},
                    {"vi": "Thời gian phân tích", "en": "Timestamp", "value": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "desc": "Thời điểm phân tích."},
                    {"vi": "Nhận xét tổng quan", "en": "Summary",
                     "value": ("Vết nứt có nguy cơ, cần kiểm tra thêm." if "Nguy hiểm" in severity else "Vết nứt nhỏ, nên tiếp tục theo dõi."),
                     "desc": "Kết luận tự động của hệ thống."},
                ]

                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("Biểu đồ thống kê")
                col_chart1, col_chart2 = st.columns(2)

                # Bar chart
                with col_chart1:
                    fig1 = plt.figure(figsize=(5, 3.2))
                    plt.bar(range(1, len(confs) + 1), confs)
                    plt.xlabel("Crack #")
                    plt.ylabel("Confidence")
                    plt.ylim(0, 1)
                    plt.title("Độ tin cậy từng vùng nứt")
                    st.pyplot(fig1)
                    bar_png = fig_to_png(fig1)
                    plt.close(fig1)

                # Pie chart
                with col_chart2:
                    labels = ["Vùng nứt lớn nhất", "Phần ảnh còn lại"]
                    sizes = [max_ratio, max(0.0, 1 - max_ratio)]
                    fig2 = plt.figure(figsize=(5, 3.2))
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
                    file_name=f"BKAI_CrackReport_{Path(uploaded_file.name).stem}.pdf",
                    mime="application/pdf",
                    key=f"pdf_btn_{idx}_{uploaded_file.name}",
                )

            with tab_stage2:
                show_stage2_demo(key_prefix=f"stage2_{idx}")


# =========================================================
# 10) ENTRY
# =========================================================
if st.session_state.authenticated:
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.username}")
        if st.button("Đăng xuất", key="btn_logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.profile_filled = False
            st.rerun()
    run_main_app()
else:
    show_auth_page()
