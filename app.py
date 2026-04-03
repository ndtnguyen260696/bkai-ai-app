import streamlit as st
import requests
from PIL import Image, ImageDraw
import colorsys
import hashlib
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
    Image as RLImage,
    Table,
    TableStyle,
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
# 0. GLOBAL CONFIGURATION
# =========================================================

A4_LANDSCAPE = landscape(A4)

ROBOFLOW_FULL_URL = os.getenv(
    "ROBOFLOW_FULL_URL",
    "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"
)

LOGO_PATH = "BKAI_Logo.png"
FONT_PATH = "times.ttf"
FONT_NAME = "TimesVN"

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
    page_title="BKAI - AI-Based Concrete Crack Detection and Classification System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 0.1 GLOBAL STYLES
# =========================================================


def inject_global_styles():
    st.markdown(
        """
        <style>
        :root{
            --bg:#edf2f8;
            --panel:#ffffff;
            --soft:#f8fbff;
            --text:#1f2a37;
            --muted:#708095;
            --line:#dbe4ef;
            --blue-1:#2b78e4;
            --blue-2:#3f8cf5;
            --blue-3:#6dc6ff;
            --accent:#ff6b6b;
            --btn:#4f9df0;
            --btn-hover:#3f8fe6;
            --btn-border:#3c8bdf;
            --shadow:0 14px 36px rgba(21, 52, 98, 0.10);
            --radius:18px;
        }

        .stApp{
            background:
                radial-gradient(circle at top left, rgba(129, 199, 255, 0.18), transparent 30%),
                linear-gradient(180deg, #eef4fb 0%, #e9eff7 100%);
        }

        html, body, [class*="css"]{
            font-family: Arial, Helvetica, sans-serif;
            color: var(--text);
        }

        .block-container{
            max-width: 1080px;
            padding-top: 0 !important;
            padding-bottom: 2rem;
        }

        div[data-testid="stAppViewContainer"] > .main {
            padding-top: 0rem;
        }

        /* ===== AUTH PAGE ===== */
        .auth-shell{
            margin: 26px auto 12px auto;
            border-radius: 22px;
            overflow: hidden;
            border: 1px solid rgba(202, 214, 228, 0.92);
            background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(255,255,255,0.90));
            box-shadow: var(--shadow);
            backdrop-filter: blur(6px);
        }

        .auth-header{
            position:relative;
            min-height:250px;
            background: linear-gradient(180deg, #3f88f2 0%, #2f78e6 48%, #2a6fdd 100%);
            border-bottom:1px solid rgba(255,255,255,0.22);
            overflow:hidden;
        }

        .auth-header::before{
            content:"";
            position:absolute;
            left:-6%;
            width:112%;
            height:78px;
            bottom:24px;
            background: rgba(152, 223, 255, 0.14);
            border-radius: 50%;
        }

        .auth-header::after{
            content:"";
            position:absolute;
            left:-10%;
            width:120%;
            height:96px;
            bottom:-22px;
            background: rgba(157, 226, 255, 0.28);
            border-radius: 50%;
        }

        .auth-header-glow{
            position:absolute;
            inset:auto -10% 40px -10%;
            height:60px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.13), transparent);
            z-index:1;
        }

        .auth-brand{
            position:relative;
            z-index:2;
            height:100%;
            display:flex;
            align-items:center;
            justify-content:center;
            padding:26px 24px 70px 24px;
        }

        .auth-logo-box{
            position:absolute;
            left:22px;
            top:22px;
            width:98px;
            height:98px;
            background:#ffffff;
            border:1px solid #d8e1eb;
            border-radius:18px;
            display:flex;
            align-items:center;
            justify-content:center;
            overflow:hidden;
            box-shadow:0 12px 26px rgba(11, 39, 84, 0.16);
        }

        .auth-brand-copy{
            text-align:center;
            color:#fff;
            max-width:640px;
        }

        .auth-overline{
            font-size:11px;
            font-weight:700;
            letter-spacing:0.18em;
            text-transform:uppercase;
            opacity:0.90;
            margin-bottom:12px;
        }

        .auth-hero-title{
            font-size:30px;
            font-weight:800;
            letter-spacing:0.02em;
            line-height:1.15;
            margin:0 0 8px 0;
        }

        .auth-hero-subtitle{
            font-size:13px;
            line-height:1.6;
            opacity:0.92;
            max-width:560px;
            margin:0 auto;
        }

        .auth-form-area{
            position:relative;
            background: linear-gradient(180deg, #f7f9fc 0%, #f1f4f8 100%);
            padding:30px 28px 30px 28px;
        }

        .auth-form-box{
            width:100%;
            max-width:520px;
            margin:0 auto;
            background:rgba(255,255,255,0.88);
            border:1px solid #e4ebf3;
            border-radius:20px;
            box-shadow:0 10px 28px rgba(34, 61, 102, 0.08);
            padding:12px 24px 22px 24px;
        }

        .portal-title{
            text-align:center;
            font-size:24px;
            font-weight:800;
            color:#1f2a37;
            margin:12px 0 24px 0;
            letter-spacing:0.01em;
        }

        .portal-caption{
            text-align:center;
            color:#7f8da0;
            font-size:12px;
            margin:4px 0 6px 0;
        }

        div[data-baseweb="tab-list"]{
            display:flex;
            justify-content:center;
            gap:28px;
            background:transparent;
            border-bottom:1px solid #e4ebf3;
            margin:0 0 18px 0;
        }

        button[data-baseweb="tab"]{
            background:transparent !important;
            border:none !important;
            border-radius:0 !important;
            padding:14px 2px 14px 2px !important;
            color:#7a8798 !important;
            font-size:13px !important;
            font-weight:600 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"]{
            color:#233144 !important;
            font-weight:700 !important;
            border-bottom:3px solid var(--accent) !important;
        }

        div[data-testid="stTextInput"] label,
        div[data-testid="stTextArea"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stFileUploader"] label{
            color:#556171 !important;
            font-size:12px !important;
            font-weight:700 !important;
        }

        div[data-testid="stTextInput"] input{
            border:none !important;
            border-bottom:1px solid #b8c3d1 !important;
            border-radius:0 !important;
            background:transparent !important;
            box-shadow:none !important;
            color:#0f1722 !important;
            padding:10px 0 !important;
            padding-left:0 !important;
            padding-right:0 !important;
            font-size:14px !important;
        }

        div[data-testid="stTextInput"] input:focus{
            border-bottom:2px solid #4f9df0 !important;
        }

        div[data-testid="stTextInput"] input::placeholder{
            color:#9aa7b6 !important;
        }

        div.stCheckbox{
            margin-top: 2px;
            margin-bottom: 10px;
        }

        div.stCheckbox label{
            color:#607084 !important;
            font-size:12px !important;
        }

        .auth-divider{
            position:relative;
            margin:22px 0 18px 0;
            height:18px;
        }

        .auth-divider::before{
            content:"";
            position:absolute;
            left:0;
            right:0;
            top:50%;
            border-top:1px solid #dfe7f0;
            transform:translateY(-50%);
        }

        .auth-divider span{
            position:absolute;
            left:50%;
            top:50%;
            transform:translate(-50%,-50%);
            background:#ffffff;
            color:#93a0af;
            font-size:11px;
            padding:0 12px;
            text-transform:lowercase;
        }

        div.stButton > button{
            width:100%;
            height:40px;
            border-radius:12px !important;
            background:linear-gradient(180deg, #5aa7f1 0%, #4697ea 100%) !important;
            border:1px solid var(--btn-border) !important;
            color:#ffffff !important;
            font-size:13px !important;
            font-weight:700 !important;
            box-shadow:0 8px 20px rgba(79, 157, 240, 0.18) !important;
        }

        div.stButton > button:hover{
            background:linear-gradient(180deg, #4e9bf0 0%, #3f8fe6 100%) !important;
            border:1px solid #3582d6 !important;
            color:#ffffff !important;
        }

        .auth-footnote{
            text-align:center;
            color:#8c98a7;
            font-size:11px;
            margin-top:12px;
            line-height:1.5;
        }

        /* regular app area */
        .bkai-main-header{
            background:
                radial-gradient(circle at right top, rgba(122, 210, 255, 0.30), transparent 30%),
                linear-gradient(135deg, #2169dd 0%, #2f79ea 55%, #4197f3 100%);
            border-radius:18px;
            padding:24px 26px;
            margin-top:18px;
            margin-bottom:20px;
            color:#ffffff;
            box-shadow:0 14px 28px rgba(44, 92, 172, 0.16);
        }

        .bkai-main-title{
            font-size:30px;
            font-weight:800;
            margin-bottom:6px;
            line-height:1.18;
        }

        .bkai-main-subtitle{
            font-size:14px;
            opacity:0.96;
            line-height:1.6;
            max-width:760px;
        }

        .bkai-card{
            background:#ffffff;
            border:1px solid #e4ebf3;
            border-radius:16px;
            padding:20px;
            box-shadow:0 8px 22px rgba(24, 48, 88, 0.05);
        }

        .bkai-status-ok{
            background:#ecfdf5;
            border:1px solid #bbf7d0;
            color:#166534;
            padding:12px 14px;
            border-radius:12px;
            font-weight:700;
        }

        .bkai-status-danger{
            background:#fef2f2;
            border:1px solid #fecaca;
            color:#991b1b;
            padding:12px 14px;
            border-radius:12px;
            font-weight:700;
        }

        .bkai-sidebar-user{
            background:#eff6ff;
            border:1px solid #bfdbfe;
            border-radius:12px;
            padding:12px 14px;
            color:#1d4ed8;
            font-weight:700;
            margin-bottom:8px;
        }

        [data-testid="stSidebar"]{
            background:#ffffff;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="select"] > div{
            border:1px solid #d6e0eb !important;
            border-radius:10px !important;
            background:#ffffff !important;
            color:#111827 !important;
        }

        div[data-testid="stDataFrame"]{
            border-radius:12px;
            overflow:hidden;
        }

        @media (max-width: 900px){
            .auth-header{
                min-height:220px;
            }
            .auth-logo-box{
                width:82px;
                height:82px;
                border-radius:14px;
            }
            .auth-brand{
                padding:88px 20px 56px 20px;
            }
            .auth-hero-title{
                font-size:22px;
            }
            .auth-form-area{
                padding:20px 14px 22px 14px;
            }
            .auth-form-box{
                max-width:100%;
                padding:10px 16px 18px 16px;
                border-radius:16px;
            }
            .bkai-main-title{
                font-size:24px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_styles()


# =========================================================
# 1. COMMON HELPERS
# =========================================================

def fig_to_png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def _hue_from_key(key: str) -> float:
    if not key:
        key = "default"
    h_int = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return (h_int % 360) / 360.0


def stable_rgb(image_key: str, instance_key: str):
    base_h = _hue_from_key("IMG|" + (image_key or "noimage"))
    inst_h = _hue_from_key("INS|" + (instance_key or "noins"))
    h = (base_h * 0.35 + inst_h * 0.65) % 1.0
    s, v = 0.90, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


@st.cache_resource
def _get_font(size=18):
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
# 1.1 POLYGON / MASK
# =========================================================

def extract_poly_points(points_field, img_w: int, img_h: int):
    pts = []

    def _append_xy(x, y):
        if x is None or y is None:
            return
        try:
            pts.append((float(x), float(y)))
        except Exception:
            return

    if isinstance(points_field, dict):
        for k in sorted(points_field.keys(), key=lambda x: str(x)):
            seg = points_field[k]
            if isinstance(seg, list):
                for p in seg:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        _append_xy(p.get("x"), p.get("y"))
                    elif isinstance(p, (list, tuple)) and len(p) == 2:
                        _append_xy(p[0], p[1])

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

    x_norm = max_x <= 1.5
    y_norm = max_y <= 1.5

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

    hist = mask.histogram()
    white = sum(hist[1:])
    return float(white)


def crack_area_ratio_percent(predictions, img_w: int, img_h: int):
    img_area = float(img_w * img_h) if img_w > 0 and img_h > 0 else 0.0
    area_px2 = crack_area_from_predictions(predictions, img_w, img_h)
    ratio = (area_px2 / img_area) if img_area > 0 else 0.0
    return (ratio * 100.0, area_px2)


# =========================================================
# 1.2 DRAW RESULT
# =========================================================

def draw_predictions_with_mask(image: Image.Image, predictions, image_key: str = "", min_conf: float = 0.0):
    base = image.convert("RGB")
    W, H = base.size

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(18)

    for i, p in enumerate(predictions):
        conf = float(p.get("confidence", 0.0))
        if conf < float(min_conf):
            continue

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

        instance_key = str(p.get("detection_id", "")).strip()
        if not instance_key:
            instance_key = f"{p.get('class','')}|{x}|{y}|{w}|{h}|{i}"

        r, g, b = stable_rgb(image_key, instance_key)

        box_color = (r, g, b, 255)
        overlay_fill = (r, g, b, 110)
        overlay_edge = (r, g, b, 255)
        text_color = (r, g, b, 255)

        pts_raw = p.get("points", None)
        poly = []
        if pts_raw is not None:
            polys = extract_polygons(pts_raw, W, H)
            if len(polys) > 0:
                poly = polys[0]

        if len(poly) >= 3:
            draw.polygon(poly, fill=overlay_fill)
            draw.line(poly + [poly[0]], fill=overlay_edge, width=1)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 60))

        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)

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

        draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=(0, 0, 0, 200))
        draw.text((lx + pad, ly + pad), label, fill=text_color, font=font)

    result = Image.alpha_composite(base.convert("RGBA"), overlay)
    return result.convert("RGB")


# =========================================================
# 1.3 SEVERITY
# =========================================================

def estimate_severity_from_ratio(area_ratio_percent: float):
    r = float(area_ratio_percent)
    if r < 0.2:
        return "Minor"
    elif r < 1.0:
        return "Moderate"
    else:
        return "Severe"


# =========================================================
# 2. PDF EXPORT
# =========================================================

def export_pdf(
    original_img,
    analyzed_img,
    metrics_df,
    chart_bar_png=None,
    chart_pie_png=None,
    filename="bkai_report_pro_plus.pdf",
):
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
        c.setFont(TITLE_FONT, TITLE_SIZE)
        c.drawCentredString(page_w / 2.0, y_top - 6 * mm, page_title)

        if subtitle:
            c.setFont(BODY_FONT, 11)
            c.drawCentredString(page_w / 2.0, y_top - 13 * mm, subtitle)

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
        summary_val = "Conclusion: Concrete cracks are present and should be further inspected."

    if "Severe" in severity_val:
        banner_fill = colors.HexColor("#ffebee")
        banner_text = colors.HexColor("#c62828")
    elif "Moderate" in severity_val:
        banner_fill = colors.HexColor("#fff3e0")
        banner_text = colors.HexColor("#ef6c00")
    else:
        banner_fill = colors.HexColor("#e8f5e9")
        banner_text = colors.HexColor("#2e7d32")

    page_no = 1
    content_top_y = draw_header("ANALYSIS REPORT", page_no=page_no)

    content_top_y -= 5 * mm
    gap_x = 10 * mm
    slot_w = (CONTENT_W - gap_x) / 2.0
    max_img_h = 90 * mm

    c.setFont(BODY_FONT, 11)
    c.setFillColor(colors.black)
    c.drawString(LEFT, content_top_y + 4 * mm, "Original Image")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Analyzed Image")

    left_bottom = draw_pil_image(original_img, LEFT, content_top_y, slot_w, max_img_h)
    right_bottom = draw_pil_image(analyzed_img, LEFT + slot_w + gap_x, content_top_y, slot_w, max_img_h)
    images_bottom_y = min(left_bottom, right_bottom)

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
        c.drawString(LEFT, bar_bottom - 10, "Confidence of each detected crack region")

    if chart_pie_png is not None:
        chart_pie_png.seek(0)
        pie_img = ImageReader(chart_pie_png)
        pw, ph = pie_img.getSize()
        scale_pie = min(chart_slot_w / pw, max_chart_h / ph)
        cw = pw * scale_pie
        ch = ph * scale_pie
        pie_bottom = charts_top_y - ch
        c.drawImage(pie_img, LEFT + chart_slot_w + gap_x, pie_bottom, width=cw, height=ch, mask="auto")
        c.setFont(BODY_FONT, 10)
        c.setFillColor(colors.black)
        c.drawString(LEFT + chart_slot_w + gap_x, pie_bottom - 10, "Crack region ratio relative to the full image")

    c.showPage()

    page_no += 1
    subtitle = "Summary table of crack-related metrics"
    content_top_y = draw_header("ANALYSIS REPORT", subtitle=subtitle, page_no=page_no)

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

    def start_table_page(pn):
        c.showPage()
        y0 = draw_header("ANALYSIS REPORT", subtitle=subtitle, page_no=pn)
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
        c.drawString(x1 + 2, top_y - header_h + 3, "Metric (VI / EN)")
        c.drawString(x2 + 2, top_y - header_h + 3, "Value")
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
        c.drawCentredString(page_w / 2, y_top - 6 * mm, "ANALYSIS REPORT")
        c.setFont(BODY_FONT, 11)
        c.drawCentredString(page_w / 2, y_top - 14 * mm, "Case: No significant crack detected")

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

    c.setFont(BODY_FONT, 11)
    c.drawString(LEFT, content_top_y + 4 * mm, "Original Image")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Analyzed Image")

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
        "No clearly visible cracks were detected in the image under the current model threshold.",
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
# 3. STAGE 2 PDF
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

    if os.path.exists(LOGO_PATH):
        logo_flow = RLImage(LOGO_PATH, width=28 * mm, height=28 * mm)
        header_table = Table(
            [[logo_flow, Paragraph("BKAI – STAGE 2 KNOWLEDGE REPORT", title_style)]],
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
        elements.append(Paragraph("BKAI – STAGE 2 KNOWLEDGE REPORT", title_style))

    elements.append(
        Paragraph(
            "Classification of common concrete cracks by structural component (beam, column, slab, wall).",
            subtitle_style,
        )
    )

    data = [[
        Paragraph("Component", normal),
        Paragraph("Crack Type", normal),
        Paragraph("Cause", normal),
        Paragraph("Shape Characteristics", normal),
        Paragraph("Illustration", normal),
    ]]

    def make_thumb(path: str):
        if isinstance(path, str) and path and os.path.exists(path):
            return RLImage(path, width=25 * mm, height=25 * mm)
        return Paragraph("—", normal)

    for _, row in component_df.iterrows():
        img_path = row.get("Image Path", "") or row.get("Illustration", "")
        data.append([
            Paragraph(str(row["Component"]), normal),
            Paragraph(str(row["Crack Type"]), normal),
            Paragraph(str(row["Cause"]), normal),
            Paragraph(str(row["Shape Characteristics"]), normal),
            make_thumb(img_path),
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
# 4. STAGE 2 VIEW
# =========================================================

def render_component_crack_table(component_df: pd.DataFrame):
    st.markdown("### 2.2. Detailed Crack Table by Structural Component")

    h1, h2, h3, h4, h5 = st.columns([1, 1.2, 2.2, 2.2, 1.6])
    header_style = (
        "background-color:#e3f2fd;padding:6px;border:1px solid #90caf9;"
        "font-weight:bold;text-align:center;color:#111827;"
    )
    h1.markdown(f"<div style='{header_style}'>Component</div>", unsafe_allow_html=True)
    h2.markdown(f"<div style='{header_style}'>Crack Type</div>", unsafe_allow_html=True)
    h3.markdown(f"<div style='{header_style}'>Cause</div>", unsafe_allow_html=True)
    h4.markdown(f"<div style='{header_style}'>Shape Characteristics</div>", unsafe_allow_html=True)
    h5.markdown(f"<div style='{header_style}'>Illustration</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:2px 0 6px 0;'>", unsafe_allow_html=True)

    for component, subdf in component_df.groupby("Component"):
        st.markdown(
            f"<div style='background-color:#bbdefb;padding:4px 10px;margin:4px 0;"
            f"font-weight:bold;border-left:4px solid #1976d2;color:#111827;'>"
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

            c2.write(row["Crack Type"])
            c3.write(row["Cause"])
            c4.write(row["Shape Characteristics"])

            img_path = row.get("Image Path", "") or row.get("Illustration", "")
            if isinstance(img_path, str) and img_path and os.path.exists(img_path):
                c5.image(img_path, use_container_width=True)
            else:
                c5.write("—")

        st.markdown("<hr style='margin:6px 0 10px 0;border-top:1px dashed #b0bec5;'>", unsafe_allow_html=True)


def show_stage2_demo(key_prefix="stage2"):
    st.subheader("Stage 2 – Crack Classification and Suggested Causes / Actions")

    st.markdown("### 2.0. Crack Classification Diagram and Structural Examples")
    col_img1, col_img2 = st.columns([3, 4])

    with col_img1:
        tree_path = "images/stage2_crack_tree.png"
        if os.path.exists(tree_path):
            st.image(tree_path, caption="Stage 2 crack classification diagram", use_container_width=True)
        else:
            st.info("Missing file: images/stage2_crack_tree.png")

    with col_img2:
        example_path = "images/stage2_structural_example.png"
        if os.path.exists(example_path):
            st.image(example_path, caption="Examples of cracks on structural components", use_container_width=True)
        else:
            st.info("Missing file: images/stage2_structural_example.png")

    st.markdown("---")

    options = [
        "I.1 Plastic Shrinkage Crack",
        "I.2 Plastic Settlement Crack",
        "II.1 Drying Shrinkage Crack",
        "II.2 Freeze–Thaw Crack",
        "II.3 Thermal Crack",
        "II.4a Chemical Crack – Sulfate Attack",
        "II.4b Chemical Crack – Alkali–Aggregate Reaction",
        "II.5 Corrosion-Induced Crack",
        "II.6a Load-Induced Crack – Flexural",
        "II.6b Load-Induced Crack – Shear / Compression / Torsion",
        "II.7 Settlement Crack",
    ]
    st.selectbox("Select a crack type (summary):", options, key=f"{key_prefix}_summary_selectbox")
    st.caption("Table 1 – Summary of crack types by failure mechanism.")

    st.subheader("Concrete Crack Classification by Structural Component")

    component_crack_data = pd.DataFrame(
        [
            {
                "Component":"Beam",
                "Crack Type":"Flexural Crack",
                "Cause":"Caused by bending moment exceeding the allowable limit; inadequate flexural reinforcement or insufficient section capacity.",
                "Shape Characteristics":"Usually appears at mid-span and is widest in the tension zone.",
                "Image Path":"images/stage2/beam_uon.png"
            },
            {
                "Component":"Beam",
                "Crack Type":"Shear Crack",
                "Cause":"High shear force; inadequate concrete shear capacity or insufficient stirrups.",
                "Shape Characteristics":"Inclined crack, often around 45° relative to the beam axis.",
                "Image Path":"images/stage2/beam_cat.png"
            },
            {
                "Component":"Beam",
                "Crack Type":"Torsional Crack",
                "Cause":"Insufficient torsional reinforcement or unsuitable cross-section design.",
                "Shape Characteristics":"Diagonal or zigzag pattern around the beam surface.",
                "Image Path":"images/stage2/beam_xoan.png"
            },
            {
                "Component":"Beam",
                "Crack Type":"Corrosion-Induced Crack",
                "Cause":"Aggressive environment, thin cover depth, and expansion due to steel corrosion.",
                "Shape Characteristics":"Runs along reinforcement lines and may be accompanied by rust staining or cover spalling.",
                "Image Path":"images/stage2/beam_anmon.png"
            },
            {
                "Component":"Column",
                "Crack Type":"Diagonal Crack",
                "Cause":"Column subjected to high combined compression, bending, or shear; insufficient material or structural capacity.",
                "Shape Characteristics":"Inclined cracks appear on the surface when the load approaches or exceeds capacity.",
                "Image Path":"images/stage2/column_cheo.png"
            },
            {
                "Component":"Column",
                "Crack Type":"Splitting / Longitudinal Crack",
                "Cause":"High compressive stress causing longitudinal splitting; weak concrete; insufficient longitudinal reinforcement.",
                "Shape Characteristics":"Multiple parallel vertical cracks.",
                "Image Path":"images/stage2/column_tach.png"
            },
            {
                "Component":"Slab",
                "Crack Type":"Plastic Shrinkage Crack",
                "Cause":"Rapid moisture evaporation while concrete is still plastic due to wind, heat, or dry conditions.",
                "Shape Characteristics":"Shallow and small cracks, often forming a polygonal pattern.",
                "Image Path":"images/stage2/slab_congot_deo.png"
            },
            {
                "Component":"Slab",
                "Crack Type":"Drying Shrinkage Crack",
                "Cause":"Shrinkage after hardening in dry or hot environments.",
                "Shape Characteristics":"Map cracking or relatively straight crack lines.",
                "Image Path":"images/stage2/slab_congot_kho.png"
            },
            {
                "Component":"Concrete Wall",
                "Crack Type":"Shrinkage Crack",
                "Cause":"Rapid moisture loss; shrinkage stress exceeds tensile capacity.",
                "Shape Characteristics":"Random, polygonal, or intersecting crack pattern.",
                "Image Path":"images/stage2/wall_congot.png"
            },
            {
                "Component":"Concrete Wall",
                "Crack Type":"Thermal Crack",
                "Cause":"Temperature difference through the wall thickness.",
                "Shape Characteristics":"Often vertical and wider in the thermal tension zone.",
                "Image Path":"images/stage2/wall_nhiet.png"
            },
        ]
    )

    render_component_crack_table(component_crack_data)
    st.caption("Table 2 – Structural component mapping with illustration examples.")

    st.markdown("### 2.3. Export Stage 2 Knowledge Report")
    csv_bytes = component_crack_data.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Download Stage 2 Table (CSV)",
        data=csv_bytes,
        file_name="BKAI_Stage2_CrackTable.csv",
        mime="text/csv",
        key=f"stage2_csv_{key_prefix}",
    )

    pdf_buf = export_stage2_pdf(component_crack_data)
    st.download_button(
        "📄 Download Stage 2 Knowledge Report (PDF)",
        data=pdf_buf.getvalue(),
        file_name="BKAI_Stage2_Report.pdf",
        mime="application/pdf",
        key=f"stage2_pdf_{key_prefix}",
    )


# =========================================================
# 5. USER STATS
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
# 6. MAIN APP
# =========================================================


def show_top_banner(username=""):
    st.markdown(
        f"""
        <div class="bkai-main-header">
            <div class="bkai-main-title">BKAI - AI-Based Concrete Crack Detection and Classification System</div>
            <div class="bkai-main-subtitle">
                {"Welcome back, " + username + ". " if username else ""}
                Upload concrete images for AI-based crack detection, segmentation, classification, and automated PDF reporting in a cleaner, more modern workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_profile_form():
    if "profile_filled" not in st.session_state:
        st.session_state.profile_filled = False

    if st.session_state.profile_filled:
        return True

    st.markdown("<div class='bkai-card'>", unsafe_allow_html=True)
    st.subheader("User Profile Information")
    st.caption("Please complete the information below before starting the analysis.")

    with st.form("user_info_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input("Full Name *", placeholder="Enter your full name")
            email = st.text_input("Email *", placeholder="Enter your email address")
            organization = st.text_input("Organization / Company", placeholder="University, company, or institution")
            occupation = st.selectbox(
                "Occupation / User Group *",
                [
                    "Student",
                    "Graduate Student / Researcher",
                    "Structural Engineer",
                    "Site Engineer",
                    "Supervision Consultant",
                    "Construction Contractor",
                    "Project Owner / Project Management",
                    "IT Engineer",
                    "Lecturer / Academic Staff",
                    "Other",
                ],
            )

        with col2:
            country = st.text_input("Country / Region", placeholder="Enter your country or region")
            project_name = st.text_input("Project / Case Name", placeholder="Optional project name")
            purpose = st.selectbox(
                "Purpose of Use",
                [
                    "Academic Research",
                    "Thesis / Dissertation",
                    "Site Inspection",
                    "Structural Monitoring",
                    "Quality Control",
                    "Training / Demonstration",
                    "Other",
                ],
            )
            notes = st.text_area("Remarks / Notes", placeholder="Optional notes", height=110)

        submit_info = st.form_submit_button("Save Profile and Start Analysis")

    if submit_info:
        if not full_name or not occupation or not email:
            st.warning("Please complete all required fields: Full Name, Occupation, and Email.")
            st.markdown("</div>", unsafe_allow_html=True)
            return False
        elif "@" not in email or "." not in email:
            st.warning("Invalid email address. Please check and try again.")
            st.markdown("</div>", unsafe_allow_html=True)
            return False
        else:
            st.session_state.profile_filled = True
            st.session_state.user_full_name = full_name
            st.session_state.user_occupation = occupation
            st.session_state.user_email = email
            st.session_state.user_organization = organization
            st.session_state.user_country = country
            st.session_state.user_project_name = project_name
            st.session_state.user_purpose = purpose
            st.session_state.user_notes = notes

            record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "login_user": st.session_state.get("username", ""),
                "full_name": full_name,
                "occupation": occupation,
                "email": email,
                "organization": organization,
                "country": country,
                "project_name": project_name,
                "purpose": purpose,
                "notes": notes,
            }
            user_stats.append(record)
            try:
                with open(USER_STATS_FILE, "w", encoding="utf-8") as f:
                    json.dump(user_stats, f, ensure_ascii=False, indent=2)
            except Exception as e:
                st.warning(f"Failed to save user statistics: {e}")

            st.success("Profile saved successfully. You can now upload images for analysis.")
            st.markdown("</div>", unsafe_allow_html=True)
            return True

    st.markdown("</div>", unsafe_allow_html=True)
    return False


def run_main_app():
    if not ROBOFLOW_FULL_URL:
        st.error("ROBOFLOW_FULL_URL is not configured.")
        st.stop()

    show_top_banner(st.session_state.get("username", ""))

    if not render_profile_form():
        return

    st.sidebar.markdown("<div class='bkai-sidebar-user'>Active User: " + st.session_state.get("username", "-") + "</div>", unsafe_allow_html=True)

    st.sidebar.header("Analysis Settings")
    min_conf = st.sidebar.slider("Minimum confidence threshold", 0.0, 1.0, 0.30, 0.05)
    st.sidebar.caption("Only crack regions with confidence greater than or equal to this threshold will be displayed.")

    with st.sidebar.expander("📊 User Statistics Manager", expanded=False):
        if user_stats:
            df_stats = pd.DataFrame(user_stats)
            st.dataframe(df_stats, use_container_width=True, height=220)
            stats_csv = df_stats.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ Download User Statistics (CSV)",
                data=stats_csv,
                file_name="BKAI_UserStats.csv",
                mime="text/csv",
            )
        else:
            st.info("No user statistics available yet.")

    st.markdown("<div class='bkai-card'>", unsafe_allow_html=True)
    st.subheader("Image Upload and Analysis")
    st.caption("Upload one or multiple concrete images. The system will perform crack detection, segmentation, and report generation.")

    uploaded_files = st.file_uploader(
        "Upload one or more concrete images (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    analyze_btn = st.button("Analyze Images")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_btn:
        if not uploaded_files:
            st.warning("Please upload at least one image before clicking Analyze Images.")
            st.stop()

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            st.write("---")
            st.markdown(f"## Image {idx}: `{uploaded_file.name}`")

            t0 = time.time()
            orig_img = Image.open(uploaded_file).convert("RGB")
            img_w, img_h = orig_img.size

            buf = io.BytesIO()
            orig_img.save(buf, format="JPEG")
            buf.seek(0)

            with st.spinner(f"Sending image {idx} to the AI inference service..."):
                try:
                    resp = requests.post(
                        ROBOFLOW_FULL_URL,
                        files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
                        timeout=60,
                    )
                except Exception as e:
                    st.error(f"Roboflow API error for image {uploaded_file.name}: {e}")
                    continue

            if resp.status_code != 200:
                st.error(f"Roboflow returned an error for image {uploaded_file.name}.")
                st.text(resp.text[:2000])
                continue

            try:
                result = resp.json()
            except Exception:
                st.error("Unable to parse JSON response from Roboflow.")
                st.text(resp.text[:2000])
                continue

            predictions = result.get("predictions", [])
            preds_conf = [p for p in predictions if float(p.get("confidence", 0)) >= float(min_conf)]

            total_time = time.time() - t0

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(orig_img, use_container_width=True)

            analyzed_img = None
            with col2:
                st.subheader("Analyzed Image")
                if len(preds_conf) == 0:
                    st.image(orig_img, use_container_width=True)
                    st.markdown(
                        "<div class='bkai-status-ok'>✅ Conclusion: No clearly visible crack was detected in this image.</div>",
                        unsafe_allow_html=True,
                    )

                    pdf_no_crack = export_pdf_no_crack(orig_img)
                    st.download_button(
                        "📄 Download PDF Report (No Crack Detected)",
                        data=pdf_no_crack.getvalue(),
                        file_name=f"BKAI_NoCrack_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf",
                        key=f"pdf_no_crack_{idx}",
                    )
                    continue
                else:
                    analyzed_img = draw_predictions_with_mask(
                        orig_img,
                        preds_conf,
                        image_key=uploaded_file.name,
                        min_conf=min_conf,
                    )
                    st.image(analyzed_img, use_container_width=True)
                    st.markdown(
                        "<div class='bkai-status-danger'>⚠️ Conclusion: Cracks were detected in this image.</div>",
                        unsafe_allow_html=True,
                    )

            st.write("---")
            tab_stage1, tab_stage2 = st.tabs(["Stage 1 – Detailed Analysis Report", "Stage 2 – Crack Classification"])

            with tab_stage1:
                st.subheader("Crack Information Table")

                confs = [float(p.get("confidence", 0)) for p in preds_conf]
                avg_conf = (sum(confs) / len(confs)) if confs else 0.0

                crack_ratio_percent, crack_area_px2 = crack_area_ratio_percent(preds_conf, img_w, img_h)
                severity = estimate_severity_from_ratio(crack_ratio_percent)

                summary_text = (
                    "Detected cracks may indicate structural concern and should be further inspected."
                    if severity == "Severe"
                    else "Detected cracks are minor or moderate; continuous monitoring is recommended."
                )

                metrics = [
                    {"vi": "Tên ảnh", "en": "Image Name", "value": uploaded_file.name, "desc": "Uploaded image filename"},
                    {"vi": "Thời gian xử lý", "en": "Total Processing Time", "value": f"{total_time:.2f} s", "desc": "Total execution time"},
                    {"vi": "Tốc độ mô hình AI", "en": "Inference Speed", "value": f"{total_time:.2f} s/image", "desc": "Processing time per image"},
                    {"vi": "Độ tin cậy trung bình", "en": "Average Confidence", "value": f"{avg_conf:.2f}", "desc": "Average confidence score"},
                    {"vi": "Diện tích vùng nứt", "en": "Crack Area (px^2)", "value": f"{crack_area_px2:.0f}", "desc": "Mask area in pixels"},
                    {"vi": "Phần trăm vùng nứt", "en": "Crack Area Ratio", "value": f"{crack_ratio_percent:.2f} %", "desc": "Crack mask area ratio"},
                    {"vi": "Chiều dài vết nứt", "en": "Crack Length", "value": "—", "desc": "Estimated if real scale is available"},
                    {"vi": "Chiều rộng vết nứt", "en": "Crack Width", "value": "—", "desc": "Requires calibrated scale"},
                    {"vi": "Mức độ nguy hiểm", "en": "Severity Level", "value": severity, "desc": "Severity estimated by crack ratio"},
                    {"vi": "Thời gian phân tích", "en": "Timestamp", "value": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "desc": "Execution timestamp"},
                    {"vi": "Nhận xét tổng quan", "en": "Summary", "value": summary_text, "desc": "Automatic system conclusion"},
                ]

                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("Statistical Charts")
                col_chart1, col_chart2 = st.columns(2)

                bar_png = None
                pie_png = None

                with col_chart1:
                    fig1, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)
                    xs = list(range(1, len(confs) + 1))
                    if xs:
                        ax.bar(xs, confs, width=0.35)
                        ax.set_ylim(0, 1)
                        ax.set_xticks(xs)
                        ax.set_xlabel("Crack #")
                        ax.set_ylabel("Confidence")
                        ax.set_title("Confidence of each crack region", pad=8)
                        ax.grid(axis="y", alpha=0.25)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)

                        for x, v in zip(xs, confs):
                            ax.text(x, min(1.0, v) + 0.02, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8)

                        if len(xs) == 1:
                            ax.set_xlim(0.5, 1.5)
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center")
                        ax.set_axis_off()

                    fig1.tight_layout()
                    st.pyplot(fig1)
                    bar_png = fig_to_png(fig1)
                    plt.close(fig1)

                with col_chart2:
                    labels = ["Crack region (mask)", "Remaining image area"]
                    ratio = crack_ratio_percent / 100.0
                    ratio = max(0.0, min(1.0, ratio))
                    sizes = [ratio, 1 - ratio]

                    fig2, ax2 = plt.subplots(figsize=(4.2, 3.2), dpi=150)
                    ax2.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
                    ax2.set_title("Crack area ratio relative to full image", pad=8)
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
                    "📄 Download PDF Report for This Image",
                    data=pdf_buf.getvalue(),
                    file_name=f"BKAI_CrackReport_{uploaded_file.name.split('.')[0]}.pdf",
                    mime="application/pdf",
                    key=f"pdf_btn_{idx}_{uploaded_file.name}",
                )

            with tab_stage2:
                show_stage2_demo(key_prefix=f"stage2_{idx}")


# =========================================================
# 7. USERS
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
    st.markdown("<div class='auth-shell'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="auth-header">
            <div class="auth-header-glow"></div>
            <div class="auth-logo-box">
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=76)
    else:
        st.markdown("<div style='font-weight:800;color:#2463d8;font-size:20px;'>BKAI</div>", unsafe_allow_html=True)

    st.markdown(
        """
            </div>
            <div class="auth-brand">
                <div class="auth-brand-copy">
                    <div class="auth-overline">BKAI Crack Analysis Portal</div>
                    <div class="auth-hero-title">AI-Based Concrete Crack Detection Platform</div>
                    <div class="auth-hero-subtitle">
                        Secure access to image-based crack detection, segmentation, reporting,
                        and structural crack classification in one integrated interface.
                    </div>
                </div>
            </div>
        </div>
        <div class="auth-form-area">
            <div class="auth-form-box">
                <div class="portal-caption">Welcome to the system</div>
        """,
        unsafe_allow_html=True,
    )

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        st.markdown("<div class='portal-title'>Sign in</div>", unsafe_allow_html=True)

        login_user = st.text_input(
            "Username",
            key="login_user",
            placeholder="Enter username"
        )
        login_pass = st.text_input(
            "Password",
            type="password",
            key="login_pass",
            placeholder="Enter password"
        )

        st.checkbox("Stay logged in", key="stay_logged_in")
        login_btn = st.button("Log in with Credentials", key="login_button")

        st.markdown("<div class='auth-divider'><span>or</span></div>", unsafe_allow_html=True)
        badge_btn = st.button("Log in with Badge", key="badge_button")

        if login_btn:
            if not login_user or not login_pass:
                st.warning("Please enter both username and password.")
            elif login_user in users and users.get(login_user) == login_pass:
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.success(f"Login successful. Welcome, {login_user}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        if badge_btn:
            st.info("Badge login is a placeholder in this demo version.")

        st.markdown(
            "<div class='auth-footnote'>Use your registered account to access the BKAI analysis workspace.</div>",
            unsafe_allow_html=True,
        )

    with tab_register:
        st.markdown("<div class='portal-title'>Create account</div>", unsafe_allow_html=True)

        reg_user = st.text_input(
            "Username",
            key="reg_user",
            placeholder="Choose a username"
        )
        reg_email = st.text_input(
            "Email",
            key="reg_email",
            placeholder="Enter your email"
        )
        reg_pass = st.text_input(
            "Password",
            type="password",
            key="reg_pass",
            placeholder="Create a password"
        )
        reg_pass2 = st.text_input(
            "Confirm Password",
            type="password",
            key="reg_pass2",
            placeholder="Re-enter password"
        )

        register_btn = st.button("Create Account", key="register_button")

        if register_btn:
            if not reg_user or not reg_email or not reg_pass or not reg_pass2:
                st.warning("Please complete all required fields.")
            elif "@" not in reg_email or "." not in reg_email:
                st.error("Please enter a valid email address.")
            elif reg_user in users:
                st.error("This username already exists. Please choose another one.")
            elif reg_pass != reg_pass2:
                st.error("Passwords do not match.")
            elif len(reg_pass) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                users[reg_user] = reg_pass
                with open(USERS_FILE, "w", encoding="utf-8") as f:
                    json.dump(users, f, ensure_ascii=False, indent=2)
                st.success("Account created successfully. You can now log in.")

        st.markdown(
            "<div class='auth-footnote'>Create an account to save your access credentials and start using the BKAI system.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div></div></div>", unsafe_allow_html=True)


# =========================================================
# 8. ENTRY
# =========================================================

if st.session_state.authenticated:
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=100)
        st.markdown(f"**User:** {st.session_state.username}")
        if st.button("Log out"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.profile_filled = False
            st.rerun()
    run_main_app()
else:
    show_auth_page()
