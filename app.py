
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
import base64
import math
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
            --blue-1:#1d64d6;
            --blue-2:#2d7df0;
            --blue-3:#5aa9ff;
            --blue-4:#87d3ff;
            --line:#d8d8d8;
            --text:#1f2937;
            --muted:#6b7280;
            --panel:#ffffff;
            --panel-2:#f8fbff;
            --shell-border:#cfd9ea;
            --shadow:0 24px 60px rgba(34,55,100,.16);
        }

        .stApp{
            background:
                radial-gradient(circle at top left, rgba(255,255,255,.96), rgba(223,231,243,.80) 22%, transparent 45%),
                linear-gradient(180deg, #e9eff8 0%, #dfe7f3 100%);
        }

        html, body, [class*="css"]{
            font-family: Inter, Arial, Helvetica, sans-serif;
            color:var(--text);
        }

        .block-container{
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"]{
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
            border-right:1px solid #d9e3f0;
        }

        .app-shell{
            border:1.5px solid var(--shell-border);
            border-radius:30px;
            overflow:hidden;
            box-shadow: var(--shadow);
            background: rgba(255,255,255,.38);
            backdrop-filter: blur(10px);
        }

        .hero{
            position:relative;
            overflow:hidden;
            padding:34px 40px 110px;
            background: linear-gradient(135deg, #4d92ff 0%, #2e6ee8 52%, #1f5bd3 100%);
        }

        .hero::before{
            content:"";
            position:absolute;
            inset:auto -10% 48px -10%;
            height:64px;
            background: rgba(255,255,255,.11);
            border-radius:50%;
        }

        .hero::after{
            content:"";
            position:absolute;
            inset:auto -10% 14px -10%;
            height:96px;
            background: rgba(130,214,255,.20);
            border-radius:50%;
        }

        .hero-flex{
            position:relative;
            z-index:2;
            display:flex;
            align-items:center;
            gap:28px;
        }

        .hero-logo{
            flex:0 0 136px;
            width:136px;
            height:136px;
            border-radius:24px;
            background:#fff;
            display:flex;
            align-items:center;
            justify-content:center;
            box-shadow: 0 16px 36px rgba(19,43,93,.28);
            border:1px solid rgba(255,255,255,.85);
            overflow:hidden;
        }

        .hero-logo-fallback{
            font-weight:800;
            font-size:30px;
            color:#2e6ee8;
            letter-spacing:.04em;
        }

        .hero-copy{color:#fff; flex:1; min-width:0;}
        .hero-kicker{
            font-size:13px;
            letter-spacing:.24em;
            text-transform:uppercase;
            opacity:.92;
            font-weight:700;
            margin-bottom:10px;
        }

        .hero-title{
            font-size:54px;
            line-height:1.05;
            font-weight:800;
            margin:0;
            letter-spacing:-0.03em;
            text-shadow:0 8px 22px rgba(0,0,0,.08);
            max-width: 860px;
        }

        .hero-subtitle{
            margin-top:16px;
            font-size:18px;
            line-height:1.7;
            color:rgba(255,255,255,.93);
            max-width:820px;
        }

        .hero-badge{
            display:inline-flex;
            align-items:center;
            justify-content:center;
            margin-top:22px;
            padding:14px 30px;
            border-radius:999px;
            color:#fff;
            font-size:22px;
            font-weight:700;
            background: rgba(255,255,255,.10);
            border:1px solid rgba(255,255,255,.18);
            backdrop-filter: blur(8px);
            box-shadow: 0 10px 30px rgba(10,40,120,.18);
        }

        .login-card-wrap{
            position:relative;
            margin-top:-74px;
            padding:0 36px 36px;
            z-index:3;
        }

        .login-card{
            max-width:960px;
            margin:0 auto;
            background: linear-gradient(180deg, rgba(255,255,255,.97) 0%, rgba(245,248,255,.96) 100%);
            border:1px solid rgba(212,223,239,.95);
            border-radius:28px;
            box-shadow: 0 25px 60px rgba(31,58,120,.18);
            overflow:hidden;
            backdrop-filter: blur(14px);
        }

        .login-card-inner{
            padding:24px 34px 32px;
        }

        div[data-baseweb="tab-list"]{
            gap:22px;
            justify-content:center;
            border-bottom:1px solid #e4eaf5;
            padding:18px 10px 0;
        }

        button[data-baseweb="tab"]{
            background:transparent !important;
            padding:12px 4px 14px !important;
            border-radius:0 !important;
            font-size:17px !important;
            font-weight:600 !important;
            color:#7080a0 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"]{
            color:#1d4ed8 !important;
            border-bottom:3px solid #2e6ee8 !important;
        }

        .section-title{
            font-size:22px;
            font-weight:800;
            color:#202a40;
            margin:10px 0 14px;
        }

        .section-note{
            font-size:14px;
            color:#7a879f;
            margin-bottom:18px;
        }

        div[data-testid="stTextInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stTextArea"] label,
        div.stCheckbox label,
        div[data-testid="stFileUploader"] label{
            font-size:14px !important;
            font-weight:600 !important;
            color:#33415c !important;
        }

        div[data-testid="stTextInput"] input{
            min-height:52px !important;
            border-radius:14px !important;
            border:1.5px solid #d8e1ef !important;
            background:#f8fbff !important;
            box-shadow:none !important;
            padding:0 16px !important;
            color:#1f2937 !important;
        }

        div[data-testid="stTextInput"] input:focus{
            border:1.5px solid #3b82f6 !important;
            background:#ffffff !important;
        }

        div.stButton > button{
            min-height:48px;
            border:none;
            border-radius:14px;
            background: linear-gradient(135deg, #4f8cff 0%, #2b6ee9 60%, #1d5ddb 100%);
            color:#fff;
            font-size:16px;
            font-weight:700;
            box-shadow: 0 16px 30px rgba(43,110,233,.18);
            transition: all .22s ease;
        }

        div.stButton > button:hover{
            transform: translateY(-1px);
            box-shadow: 0 20px 36px rgba(43,110,233,.25);
        }

        .secondary-btn button{
            background: linear-gradient(135deg, #66b9ff 0%, #4a95ff 50%, #5bc1ff 100%) !important;
        }

        .form-sep{
            display:flex;
            align-items:center;
            gap:12px;
            margin:18px 0 14px;
            color:#93a1ba;
            font-size:12px;
            text-transform:uppercase;
            letter-spacing:.18em;
        }

        .form-sep::before,
        .form-sep::after{
            content:"";
            flex:1;
            height:1px;
            background:#e1e8f3;
        }

        .help-note{
            margin-top:18px;
            text-align:center;
            font-size:13px;
            color:#7a879f;
        }

        .bkai-main-header{
            background: linear-gradient(135deg, #4f8cff 0%, #2563eb 55%, #1e4fc4 100%);
            border-radius:24px;
            padding:24px 28px;
            margin-top:8px;
            margin-bottom:18px;
            color:#ffffff;
            box-shadow: var(--shadow);
        }

        .bkai-main-title{
            font-size:34px;
            font-weight:800;
            margin-bottom:8px;
        }

        .bkai-main-subtitle{
            font-size:16px;
            opacity:0.95;
            max-width:820px;
            line-height:1.7;
        }

        .bkai-card{
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            border:1px solid #dde6f2;
            border-radius:24px;
            padding:22px;
            box-shadow: 0 16px 34px rgba(31,58,120,.08);
            margin-bottom:16px;
        }

        .bkai-status-ok{
            background:#ebfaf0;
            border:1px solid #c7efda;
            color:#198754;
            padding:12px 14px;
            border-radius:14px;
            font-weight:700;
        }

        .bkai-status-danger{
            background:#fef2f2;
            border:1px solid #fecaca;
            color:#991b1b;
            padding:12px 14px;
            border-radius:14px;
            font-weight:700;
        }

        .bkai-sidebar-user{
            background:#eff6ff;
            border:1px solid #bfdbfe;
            border-radius:12px;
            padding:10px 12px;
            color:#1d4ed8;
            font-weight:700;
            margin-bottom:8px;
        }

        .metric-box{
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbe7f5;
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 24px rgba(31,58,120,.08);
            min-height: 120px;
            margin-bottom: 12px;
        }

        .metric-box.metric-minor{
            border: 1px solid #bfe7cd;
            background: linear-gradient(180deg, #f4fff7 0%, #ecfbf1 100%);
        }

        .metric-box.metric-moderate{
            border: 1px solid #ffd59c;
            background: linear-gradient(180deg, #fffaf2 0%, #fff3df 100%);
        }

        .metric-box.metric-severe{
            border: 1px solid #f3b4b4;
            background: linear-gradient(180deg, #fff6f6 0%, #ffebeb 100%);
        }

        .metric-name{
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: #64748b;
            margin-bottom: 8px;
        }

        .metric-number{
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.2;
            margin-bottom: 6px;
            word-break: break-word;
        }

        .metric-help{
            font-size: 13px;
            line-height: 1.5;
            color: #64748b;
        }

        .metric-summary{
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbe7f5;
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 10px 24px rgba(31,58,120,.08);
            margin-top: 8px;
            margin-bottom: 12px;
        }

        .metric-summary.metric-minor{
            border: 1px solid #bfe7cd;
            background: linear-gradient(180deg, #f4fff7 0%, #ecfbf1 100%);
        }

        .metric-summary.metric-moderate{
            border: 1px solid #ffd59c;
            background: linear-gradient(180deg, #fffaf2 0%, #fff3df 100%);
        }

        .metric-summary.metric-severe{
            border: 1px solid #f3b4b4;
            background: linear-gradient(180deg, #fff6f6 0%, #ffebeb 100%);
        }

        .metric-summary-title{
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: #64748b;
            margin-bottom: 8px;
        }

        .metric-summary-text{
            font-size: 18px;
            font-weight: 700;
            line-height: 1.6;
            color: #0f172a;
        }

        .stage2-wrap{
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #d9e6f5;
            border-radius: 24px;
            padding: 22px 22px 18px 22px;
            box-shadow: 0 16px 34px rgba(31,58,120,.08);
            margin-top: 10px;
            margin-bottom: 18px;
        }

        .stage2-title{
            font-size: 30px;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 8px;
        }

        .stage2-subtitle{
            font-size: 15px;
            line-height: 1.7;
            color: #64748b;
            margin-bottom: 18px;
            max-width: 920px;
        }

        .stage2-section-box{
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            border: 1px solid #d8e5f3;
            border-radius: 20px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 10px 24px rgba(31,58,120,.06);
            margin-bottom: 18px;
        }

        .stage2-section-title{
            font-size: 22px;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 8px;
        }

        .stage2-section-note{
            font-size: 14px;
            line-height: 1.6;
            color: #64748b;
            margin-bottom: 14px;
        }

        .stage2-component-block{
            margin-top: 14px;
            margin-bottom: 18px;
            border: 1px solid #d9e6f5;
            border-radius: 18px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: 0 8px 20px rgba(31,58,120,.05);
        }

        .stage2-component-header{
            background: linear-gradient(135deg, #4f8cff 0%, #2b6ee9 60%, #1d5ddb 100%);
            color: white;
            padding: 12px 16px;
            font-size: 16px;
            font-weight: 800;
            letter-spacing: .04em;
            text-transform: uppercase;
        }

        .stage2-table-header{
            display:grid;
            grid-template-columns: 1.15fr 1.45fr 2.4fr 2.4fr 1.45fr;
            gap: 0;
            background: #eaf4ff;
            border-bottom: 1px solid #d6e5f5;
        }

        .stage2-table-header div{
            padding: 12px 12px;
            font-size: 13px;
            font-weight: 800;
            color: #1e293b;
            border-right: 1px solid #d6e5f5;
        }

        .stage2-table-header div:last-child{ border-right:none; }

        .stage2-row{
            display:grid;
            grid-template-columns: 1.15fr 1.45fr 2.4fr 2.4fr 1.45fr;
            gap:0;
            border-bottom:1px solid #e6edf6;
            align-items:stretch;
            background:#ffffff;
        }

        .stage2-row:nth-child(even){ background:#fbfdff; }

        .stage2-cell{
            padding:14px 12px;
            border-right:1px solid #e6edf6;
            font-size:14px;
            line-height:1.7;
            color:#334155;
            display:flex;
            align-items:flex-start;
            min-height:132px;
        }

        .stage2-cell:last-child{ border-right:none; }
        .stage2-cell.component-name{ font-weight:800; color:#1e293b; }
        .stage2-cell.crack-name{ font-weight:700; color:#2563eb; }

        .stage2-imgbox{
            width:100%;
            min-height:104px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border:1px solid #dbe6f3;
            border-radius:14px;
            display:flex;
            align-items:center;
            justify-content:center;
            padding:8px;
        }

        .stage2-imgbox img{
            max-width:100%;
            max-height:96px;
            object-fit:contain;
            display:block;
            border-radius:8px;
        }

        .stage2-empty{
            color:#94a3b8;
            font-style:italic;
            font-size:13px;
        }

        @media (max-width: 980px){
            .stage2-table-header,
            .stage2-row{ grid-template-columns:1fr; }

            .stage2-table-header div,
            .stage2-cell{
                border-right:none;
                border-bottom:1px solid #e6edf6;
                min-height:auto;
            }
            .stage2-cell{ padding:10px 12px; }
            .stage2-imgbox{ min-height:90px; }
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
            draw.line(poly + [poly[0]], fill=overlay_edge, width=2)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 60))

        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=3)

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
# 1.3 GEOMETRY MEASUREMENT WITHOUT SCIPY / SKIMAGE
# =========================================================

def build_union_mask_from_predictions(predictions, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for p in predictions:
        pts_raw = p.get("points", None)
        if pts_raw is None:
            continue

        polys = []
        if isinstance(pts_raw, dict):
            for k in sorted(pts_raw.keys(), key=lambda z: str(z)):
                seg = pts_raw[k]
                poly = []
                if isinstance(seg, list):
                    for pt in seg:
                        if isinstance(pt, dict) and "x" in pt and "y" in pt:
                            poly.append([int(float(pt["x"])), int(float(pt["y"]))])
                        elif isinstance(pt, (list, tuple)) and len(pt) == 2:
                            poly.append([int(float(pt[0])), int(float(pt[1]))])
                if len(poly) >= 3:
                    polys.append(np.array(poly, dtype=np.int32))
        elif isinstance(pts_raw, list):
            poly = []
            for pt in pts_raw:
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    poly.append([int(float(pt["x"])), int(float(pt["y"]))])
                elif isinstance(pt, (list, tuple)) and len(pt) == 2:
                    poly.append([int(float(pt[0])), int(float(pt[1]))])
            if len(poly) >= 3:
                polys.append(np.array(poly, dtype=np.int32))

        for poly in polys:
            cv2.fillPoly(mask, [poly], 255)

    return mask

def morphological_skeleton(binary_mask):
    thin = binary_mask.copy()
    skel = np.zeros_like(thin)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(thin, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(thin, opened)
        skel = cv2.bitwise_or(skel, temp)
        thin = eroded.copy()
        if cv2.countNonZero(thin) == 0:
            break
    return skel

def measure_crack_geometry_from_mask(mask):
    if mask is None or mask.size == 0 or cv2.countNonZero(mask) == 0:
        return {
            "length_px": 0.0,
            "avg_width_px": 0.0,
            "max_width_px": 0.0,
            "skeleton": np.zeros_like(mask),
            "dist_map": np.zeros(mask.shape, dtype=np.float32),
            "max_point": (0, 0),
        }

    skeleton = morphological_skeleton(mask)
    length_px = float(cv2.countNonZero(skeleton))
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    skeleton_values = dist_map[skeleton > 0]
    if len(skeleton_values) > 0:
        avg_width_px = float(np.mean(skeleton_values) * 2.0)
    else:
        avg_width_px = 0.0

    _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
    max_width_px = float(max_val * 2.0)

    return {
        "length_px": length_px,
        "avg_width_px": avg_width_px,
        "max_width_px": max_width_px,
        "skeleton": skeleton,
        "dist_map": dist_map,
        "max_point": max_loc,
    }

def create_measurement_visualization(
    analyzed_pil_image,
    predictions,
    length_value,
    avg_width_value,
    max_width_value,
    unit_text="px",
    panel_ratio=0.36,
):
    img_rgb = np.array(analyzed_pil_image.convert("RGB"))
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    left_w = int(w * (1.0 - panel_ratio))
    right_w = w - left_w
    crack_view = img.copy()

    box_color = (118, 52, 255)
    mask_fill = (145, 88, 255)
    center_color = (40, 195, 255)
    width_line_color = (90, 110, 255)
    point_color = (255, 255, 255)

    union_mask = np.zeros((h, w), dtype=np.uint8)
    first_label_done = False

    for p in predictions:
        x = p.get("x")
        y = p.get("y")
        bw = p.get("width")
        bh = p.get("height")
        if None in (x, y, bw, bh):
            continue

        x0 = max(0, int(float(x) - float(bw) / 2))
        y0 = max(0, int(float(y) - float(bh) / 2))
        x1 = min(w - 1, int(float(x) + float(bw) / 2))
        y1 = min(h - 1, int(float(y) + float(bh) / 2))
        cv2.rectangle(crack_view, (x0, y0), (x1, y1), box_color, 3)

        pts_raw = p.get("points", None)
        polys = []
        if isinstance(pts_raw, dict):
            for k in sorted(pts_raw.keys(), key=lambda z: str(z)):
                seg = pts_raw[k]
                poly = []
                if isinstance(seg, list):
                    for pt in seg:
                        if isinstance(pt, dict) and "x" in pt and "y" in pt:
                            poly.append([int(float(pt["x"])), int(float(pt["y"]))])
                        elif isinstance(pt, (list, tuple)) and len(pt) == 2:
                            poly.append([int(float(pt[0])), int(float(pt[1]))])
                if len(poly) >= 3:
                    polys.append(np.array(poly, dtype=np.int32))
        elif isinstance(pts_raw, list):
            poly = []
            for pt in pts_raw:
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    poly.append([int(float(pt["x"])), int(float(pt["y"]))])
                elif isinstance(pt, (list, tuple)) and len(pt) == 2:
                    poly.append([int(float(pt[0])), int(float(pt[1]))])
            if len(poly) >= 3:
                polys.append(np.array(poly, dtype=np.int32))

        if polys:
            overlay = crack_view.copy()
            cv2.fillPoly(overlay, polys, mask_fill)
            crack_view = cv2.addWeighted(overlay, 0.34, crack_view, 0.66, 0)
            for poly in polys:
                cv2.polylines(crack_view, [poly], True, box_color, 3)
                cv2.fillPoly(union_mask, [poly], 255)

        if not first_label_done:
            conf = float(p.get("confidence", 0.0))
            label = f"crack {conf:.2f}"
            cv2.rectangle(crack_view, (10, 10), (128, 46), box_color, -1)
            cv2.putText(
                crack_view, label, (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            first_label_done = True

    if cv2.countNonZero(union_mask) > 0:
        skeleton = morphological_skeleton(union_mask)
        crack_view[skeleton > 0] = center_color

        dist_map = cv2.distanceTransform(union_mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
        cx, cy = int(max_loc[0]), int(max_loc[1])
        radius = int(max_val)

        x0 = max(0, cx - radius)
        x1 = min(w - 1, cx + radius)
        cv2.line(crack_view, (x0, cy), (x1, cy), width_line_color, 3)
        cv2.circle(crack_view, (cx, cy), 5, point_color, -1)
        cv2.circle(crack_view, (cx, cy), 8, width_line_color, 2)

    crack_view = cv2.resize(crack_view, (left_w, h))

    panel = np.zeros((h, right_w, 3), dtype=np.uint8)
    panel[:] = (58, 86, 132)

    title_fs = 0.54
    value_fs = 0.88
    title_th = 1
    value_th = 2

    rows_y = [int(h * 0.22), int(h * 0.50), int(h * 0.78)]
    labels = ["Length", "Avg Width", "Max Width"]
    values = [
        f"{length_value:.1f} {unit_text}",
        f"{avg_width_value:.2f} {unit_text}",
        f"{max_width_value:.2f} {unit_text}",
    ]
    colors_row = [
        (255, 170, 80),
        (120, 220, 120),
        (110, 120, 255),
    ]

    for i, ymid in enumerate(rows_y):
        c = colors_row[i]
        cv2.line(panel, (24, ymid - 8), (56, ymid - 8), c, 3)
        cv2.circle(panel, (24, ymid - 8), 4, c, -1)
        cv2.circle(panel, (56, ymid - 8), 4, c, -1)

        cv2.putText(
            panel, labels[i], (78, ymid - 8),
            cv2.FONT_HERSHEY_SIMPLEX, title_fs,
            (255, 255, 255), title_th, cv2.LINE_AA
        )
        cv2.putText(
            panel, values[i], (78, ymid + 22),
            cv2.FONT_HERSHEY_SIMPLEX, value_fs,
            c, value_th, cv2.LINE_AA
        )

        if i < 2:
            cv2.line(panel, (18, ymid + 46), (right_w - 18, ymid + 46), (108, 125, 156), 1)

    cv2.rectangle(panel, (0, 0), (right_w - 1, h - 1), (72, 98, 145), 2)
    final = np.concatenate([crack_view, panel], axis=1)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final)


# =========================================================
# 1.4 SEVERITY
# =========================================================

def estimate_severity_from_ratio(area_ratio_percent: float):
    r = float(area_ratio_percent)
    if r < 0.2:
        return "Minor"
    elif r < 1.0:
        return "Moderate"
    return "Severe"


# =========================================================
# 2. METRICS DASHBOARD
# =========================================================

def render_metrics_dashboard(metrics_df: pd.DataFrame):
    if metrics_df is None or metrics_df.empty:
        st.info("No metrics available.")
        return

    severity_value = ""
    summary_value = ""
    summary_desc = ""
    normal_rows = []

    for _, row in metrics_df.iterrows():
        metric = str(row.get("Metric", "")).strip()
        value = str(row.get("Value", "")).strip()
        desc = str(row.get("Description", "")).strip()

        if metric == "Severity Level":
            severity_value = value
        elif metric == "Summary":
            summary_value = value
            summary_desc = desc
        else:
            normal_rows.append({"Metric": metric, "Value": value, "Description": desc})

    severity_class = ""
    sv = severity_value.lower()
    if "minor" in sv:
        severity_class = "metric-minor"
    elif "moderate" in sv:
        severity_class = "metric-moderate"
    elif "severe" in sv:
        severity_class = "metric-severe"

    cols_per_row = 3
    for i in range(0, len(normal_rows), cols_per_row):
        row_items = normal_rows[i:i + cols_per_row]
        cols = st.columns(cols_per_row)

        for j in range(cols_per_row):
            with cols[j]:
                if j < len(row_items):
                    item = row_items[j]
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-name">{item['Metric']}</div>
                            <div class="metric-number">{item['Value']}</div>
                            <div class="metric-help">{item['Description']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.empty()

    if severity_value:
        st.markdown(
            f"""
            <div class="metric-summary {severity_class}">
                <div class="metric-summary-title">Severity Level</div>
                <div class="metric-summary-text">{severity_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if summary_value:
        st.markdown(
            f"""
            <div class="metric-summary {severity_class}">
                <div class="metric-summary-title">Summary</div>
                <div class="metric-summary-text">{summary_value}</div>
                <div class="metric-help" style="margin-top:8px;">{summary_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 3. PDF EXPORT
# =========================================================

def export_pdf(
    original_img,
    analyzed_img,
    metrics_df,
    chart_bar_png=None,
    chart_pie_png=None,
    measurement_visual_img=None,
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

    def wrap_text(text, font_name, font_size, max_width):
        words = str(text).split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for w_ in words[1:]:
            trial = current + " " + w_
            if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = w_
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
        return leading * len(lines) + 4

    severity_val = ""
    summary_val = ""
    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            metric = str(row.get("Metric", "")).strip()
            if metric == "Severity Level":
                severity_val = str(row.get("Value", ""))
            if metric == "Summary":
                summary_val = str(row.get("Value", ""))

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

    # PAGE 1
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
    banner_bottom = max(BOTTOM + 40 * mm, images_bottom_y - 12 * mm)

    c.setFillColor(banner_fill)
    c.setStrokeColor(colors.transparent)
    c.rect(LEFT, banner_bottom, CONTENT_W, banner_h, stroke=0, fill=1)
    c.setFillColor(banner_text)
    c.setFont(BODY_FONT, 11)
    c.drawString(LEFT + 4 * mm, banner_bottom + banner_h / 2.0 - 4, summary_val)

    charts_top_y = banner_bottom - 18 * mm
    max_chart_h = 62 * mm
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

    # PAGE 2
    page_no += 1
    content_top_y = draw_header("ANALYSIS REPORT", subtitle="Annotated measurement view and metrics summary", page_no=page_no)

    if measurement_visual_img is not None:
        c.setFont(BODY_FONT, 11)
        c.setFillColor(colors.black)
        c.drawString(LEFT, content_top_y + 4 * mm, "Annotated Measurement View")
        img = ImageReader(measurement_visual_img)
        iw, ih = img.getSize()
        max_w = CONTENT_W
        max_h = 70 * mm
        scale = min(max_w / iw, max_h / ih, 1.0)
        draw_w = iw * scale
        draw_h = ih * scale
        y_bottom = content_top_y - draw_h
        c.drawImage(img, LEFT, y_bottom, width=draw_w, height=draw_h, mask="auto")
        content_top_y = y_bottom - 12 * mm

    rows = []
    skip_keys = {"Summary"}
    if metrics_df is not None:
        for _, r in metrics_df.iterrows():
            name = str(r.get("Metric", "")).strip()
            if name in skip_keys:
                continue
            rows.append((name, str(r.get("Value", ""))))

    col1_w = 12 * mm
    col2_w = 95 * mm
    col3_w = CONTENT_W - col1_w - col2_w
    header_h = 10 * mm
    base_lead = 4.0

    def start_table_page(pn):
        c.showPage()
        y0 = draw_header("ANALYSIS REPORT", subtitle="Metrics summary", page_no=pn)
        return y0 - 10 * mm

    x0 = LEFT
    x1 = x0 + col1_w
    x2 = x1 + col2_w

    def draw_table_header(top_y):
        c.setFillColor(colors.HexColor("#1e88e5"))
        c.rect(x0, top_y - header_h, CONTENT_W, header_h, stroke=0, fill=1)
        c.setFont(BODY_FONT, 10)
        c.setFillColor(colors.white)
        c.drawString(x0 + 2, top_y - header_h + 3, "No.")
        c.drawString(x1 + 2, top_y - header_h + 3, "Metric")
        c.drawString(x2 + 2, top_y - header_h + 3, "Value")
        return top_y - header_h

    current_y = draw_table_header(content_top_y)
    for i, (label, val) in enumerate(rows, start=1):
        label_lines = wrap_text(label, BODY_FONT, BODY_SIZE, col2_w - 4)
        value_lines = wrap_text(val, BODY_FONT, BODY_SIZE, col3_w - 4)
        n_lines = max(len(label_lines), len(value_lines))
        leading = BODY_SIZE + base_lead
        row_h = n_lines * leading + 6

        if current_y - row_h < BOTTOM + 25 * mm:
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

    def draw_header():
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
        c.drawCentredString(page_w / 2, y_top - 6 * mm, "ANALYSIS REPORT")
        c.setFont(FONT_NAME, 11)
        c.drawCentredString(page_w / 2, y_top - 14 * mm, "Case: No significant crack detected")
        return y_top - max(logo_h, 15 * mm) - 20 * mm

    content_top_y = draw_header()
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
    c.drawString(LEFT, content_top_y + 4 * mm, "Original Image")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Analyzed Image")
    left_bottom = draw_pil(original_img, LEFT, content_top_y)
    draw_pil(original_img, LEFT + slot_w + gap_x, content_top_y)

    banner_y = left_bottom - 12 * mm
    banner_h = 16 * mm
    c.setFillColor(colors.HexColor("#e8f5e9"))
    c.rect(LEFT, banner_y, CONTENT_W, banner_h, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#2e7d32"))
    c.setFont(FONT_NAME, 11)
    c.drawString(
        LEFT + 4 * mm,
        banner_y + banner_h / 2 - 4,
        "No clearly visible cracks were detected in the image under the current model threshold.",
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
# 4. STAGE 2 PDF
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
        "TitleStage2", parent=styles["Title"], fontName=FONT_NAME,
        alignment=1, fontSize=18, leading=22, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "SubTitleStage2", parent=styles["Normal"], fontName=FONT_NAME,
        alignment=1, fontSize=10, leading=12, textColor=colors.grey, spaceAfter=8
    )
    normal = ParagraphStyle(
        "NormalStage2", parent=styles["Normal"], fontName=FONT_NAME,
        fontSize=8, leading=10
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
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("GRID", (0, 0), (-1, -1), 0, colors.white),
            ])
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
        TableStyle([
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
        ])
    )
    elements.append(table)
    doc.build(elements)
    buf.seek(0)
    return buf


# =========================================================
# 5. STAGE 2 VIEW
# =========================================================

def render_component_crack_table(component_df: pd.DataFrame):
    if component_df is None or component_df.empty:
        st.info("No Stage 2 crack classification data available.")
        return

    st.markdown(
        """
        <div class="stage2-section-box">
            <div class="stage2-section-title">2.2. Detailed Crack Table by Structural Component</div>
            <div class="stage2-section-note">
                The following knowledge table summarizes typical crack types by structural component,
                including possible causes, visual characteristics, and illustration images for quick reference.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_header_html = """
    <div class="stage2-table-header">
        <div>Component</div>
        <div>Crack Type</div>
        <div>Cause</div>
        <div>Shape Characteristics</div>
        <div>Illustration</div>
    </div>
    """

    grouped = component_df.groupby("Component", sort=False)
    for component, subdf in grouped:
        html = f"""
        <div class="stage2-component-block">
            <div class="stage2-component-header">{str(component).upper()}</div>
            {table_header_html}
        """

        first_row = True
        for _, row in subdf.iterrows():
            component_name = str(row.get("Component", "")).strip()
            crack_type = str(row.get("Crack Type", "")).strip()
            cause = str(row.get("Cause", "")).strip()
            shape = str(row.get("Shape Characteristics", "")).strip()
            img_path = str(row.get("Image Path", "") or row.get("Illustration", "")).strip()

            component_cell = component_name if first_row else ""
            first_row = False

            img_html = '<div class="stage2-empty">No image</div>'
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                img_html = f'<img src="data:image/png;base64,{img_b64}" alt="{crack_type}" />'

            html += f"""
            <div class="stage2-row">
                <div class="stage2-cell component-name">{component_cell}</div>
                <div class="stage2-cell crack-name">{crack_type}</div>
                <div class="stage2-cell">{cause}</div>
                <div class="stage2-cell">{shape}</div>
                <div class="stage2-cell">
                    <div class="stage2-imgbox">{img_html}</div>
                </div>
            </div>
            """
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

def show_stage2_demo(key_prefix="stage2"):
    st.markdown(
        """
        <div class="stage2-wrap">
            <div class="stage2-title">Concrete Crack Classification by Structural Component</div>
            <div class="stage2-subtitle">
                This section provides a structured knowledge base of common concrete crack types
                grouped by structural component, together with typical causes, geometric shape
                characteristics, and illustration examples for engineering interpretation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stage2-section-box">
            <div class="stage2-section-title">2.0. Crack Classification Diagram and Structural Examples</div>
            <div class="stage2-section-note">
                The two panels below present the general crack classification framework and a set of structural examples
                to support quick interpretation before reviewing the detailed table.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown(
        """
        <div class="stage2-section-box">
            <div class="stage2-section-title">2.1. Summary of Crack Types by Failure Mechanism</div>
            <div class="stage2-section-note">
                Select a representative crack category to review the overall failure mechanism classification before
                moving to the component-based crack table.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    component_crack_data = pd.DataFrame([
        {"Component":"Beam","Crack Type":"Flexural Crack","Cause":"Caused by bending moment exceeding the allowable limit; inadequate flexural reinforcement or insufficient section capacity.","Shape Characteristics":"Usually appears at mid-span and is widest in the tension zone.","Image Path":"images/stage2/beam_uon.png"},
        {"Component":"Beam","Crack Type":"Shear Crack","Cause":"High shear force; inadequate concrete shear capacity or insufficient stirrups.","Shape Characteristics":"Inclined crack, often around 45° relative to the beam axis.","Image Path":"images/stage2/beam_cat.png"},
        {"Component":"Beam","Crack Type":"Torsional Crack","Cause":"Insufficient torsional reinforcement or unsuitable cross-section design.","Shape Characteristics":"Diagonal or zigzag pattern around the beam surface.","Image Path":"images/stage2/beam_xoan.png"},
        {"Component":"Beam","Crack Type":"Tensile Crack","Cause":"Direct tensile stress exceeds the tensile strength of concrete.","Shape Characteristics":"Mostly vertical cracks distributed within the tension zone.","Image Path":"images/stage2/beam_keo.png"},
        {"Component":"Beam","Crack Type":"Sliding Crack","Cause":"Weak bonding or sliding along an interface in the member.","Shape Characteristics":"Long horizontal crack near the interface zone.","Image Path":"images/stage2/beam_truot.png"},
        {"Component":"Beam","Crack Type":"Corrosion-Induced Crack","Cause":"Aggressive environment, thin cover depth, and expansion due to steel corrosion.","Shape Characteristics":"Runs along reinforcement lines and may be accompanied by rust staining or cover spalling.","Image Path":"images/stage2/beam_anmon.png"},
        {"Component":"Column","Crack Type":"Diagonal Crack","Cause":"Column subjected to high combined compression, bending, or shear; insufficient material or structural capacity.","Shape Characteristics":"Inclined cracks appear on the surface when the load approaches or exceeds capacity.","Image Path":"images/stage2/column_cheo.png"},
        {"Component":"Column","Crack Type":"Horizontal Crack","Cause":"Transverse tensile stress or confinement-related failure under loading.","Shape Characteristics":"Horizontal cracks crossing the column width.","Image Path":"images/stage2/column_ngang.png"},
        {"Component":"Column","Crack Type":"Shrinkage Crack","Cause":"Volume reduction caused by moisture loss after hardening.","Shape Characteristics":"Fine random cracks distributed on the concrete surface.","Image Path":"images/stage2/column_congot.png"},
        {"Component":"Column","Crack Type":"Splitting / Longitudinal Crack","Cause":"High compressive stress causing longitudinal splitting; weak concrete; insufficient longitudinal reinforcement.","Shape Characteristics":"Multiple parallel vertical cracks.","Image Path":"images/stage2/column_tach.png"},
        {"Component":"Column","Crack Type":"Corrosion-Induced Crack","Cause":"Expansion of reinforcement due to corrosion in aggressive exposure conditions.","Shape Characteristics":"Vertical cracks along reinforcement lines, often accompanied by rust staining.","Image Path":"images/stage2/column_anmon.png"},
        {"Component":"Slab","Crack Type":"Plastic Shrinkage Crack","Cause":"Rapid moisture evaporation while concrete is still plastic due to wind, heat, or dry conditions.","Shape Characteristics":"Shallow and small cracks, often forming a polygonal pattern.","Image Path":"images/stage2/slab_congot_deo.png"},
        {"Component":"Slab","Crack Type":"Drying Shrinkage Crack","Cause":"Shrinkage after hardening in dry or hot environments.","Shape Characteristics":"Map cracking or relatively straight crack lines.","Image Path":"images/stage2/slab_congot_kho.png"},
        {"Component":"Slab","Crack Type":"Thermal Crack","Cause":"Temperature variation and restraint within the slab.","Shape Characteristics":"One or more dominant cracks, often wider at the exposed surface.","Image Path":"images/stage2/slab_nhiet.png"},
        {"Component":"Slab","Crack Type":"Flexural Crack","Cause":"Bending stress exceeds the tensile capacity of the slab.","Shape Characteristics":"Cracks develop in the tension zone, usually initiating from the bottom surface.","Image Path":"images/stage2/slab_uon.png"},
        {"Component":"Slab","Crack Type":"Shear Crack","Cause":"Punching or local shear stress exceeds slab resistance.","Shape Characteristics":"Inclined cracks or local failure zones near concentrated loads.","Image Path":"images/stage2/slab_cat.png"},
        {"Component":"Slab","Crack Type":"Torsional Crack","Cause":"Torsional action or restraint at slab edges and corners.","Shape Characteristics":"Diagonal or twisting crack pattern near slab corners or edge regions.","Image Path":"images/stage2/slab_xoan.png"},
        {"Component":"Slab","Crack Type":"Concentrated Load Crack","Cause":"Local stress concentration under point loading.","Shape Characteristics":"Radial cracks spreading from the loaded area.","Image Path":"images/stage2/slab_taptrung.png"},
        {"Component":"Slab","Crack Type":"Distributed Load Crack","Cause":"Distributed service loads causing flexural distress over a wider panel area.","Shape Characteristics":"Multiple cracks distributed across the slab panel.","Image Path":"images/stage2/slab_phanbo.png"},
        {"Component":"Slab","Crack Type":"Corrosion-Induced Crack","Cause":"Steel corrosion and expansion of reinforcement within the slab cover zone.","Shape Characteristics":"Cracks follow reinforcement layout and may lead to cover delamination.","Image Path":"images/stage2/slab_anmon.png"},
        {"Component":"Concrete Wall","Crack Type":"Shrinkage Crack","Cause":"Rapid moisture loss; shrinkage stress exceeds tensile capacity.","Shape Characteristics":"Random, polygonal, or intersecting crack pattern.","Image Path":"images/stage2/wall_congot.png"},
        {"Component":"Concrete Wall","Crack Type":"Thermal Crack","Cause":"Temperature difference through the wall thickness.","Shape Characteristics":"Often vertical and wider in the thermal tension zone.","Image Path":"images/stage2/wall_nhiet.png"},
        {"Component":"Concrete Wall","Crack Type":"Vertical Load Crack","Cause":"Axial or gravity load exceeds local tensile resistance in the wall body.","Shape Characteristics":"Mostly vertical cracks extending along the wall height.","Image Path":"images/stage2/wall_doc_taitrong.png"},
        {"Component":"Concrete Wall","Crack Type":"Horizontal Load Crack","Cause":"Lateral action or bending causes horizontal tensile zones in the wall.","Shape Characteristics":"Horizontal cracking across the wall face.","Image Path":"images/stage2/wall_ngang_taitrong.png"},
        {"Component":"Concrete Wall","Crack Type":"Diagonal Load Crack","Cause":"Combined shear and bending due to lateral loading or restraint.","Shape Characteristics":"Inclined diagonal cracking across the wall panel.","Image Path":"images/stage2/wall_cheo_taitrong.png"},
        {"Component":"Concrete Wall","Crack Type":"Corrosion-Induced Crack","Cause":"Corrosion of embedded reinforcement causing expansion and internal tensile stress.","Shape Characteristics":"Longitudinal cracking following reinforcement positions.","Image Path":"images/stage2/wall_anmon.png"},
    ])

    render_component_crack_table(component_crack_data)
    st.caption("Table 2 – Structural component mapping with illustration examples.")

    st.markdown(
        """
        <div class="stage2-section-box">
            <div class="stage2-section-title">2.3. Export Stage 2 Knowledge Report</div>
            <div class="stage2-section-note">
                Download the current Stage 2 knowledge table as CSV or PDF for documentation, presentation,
                or engineering reference.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    csv_bytes = component_crack_data.to_csv(index=False).encode("utf-8-sig")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            "⬇ Download Stage 2 Table (CSV)",
            data=csv_bytes,
            file_name="BKAI_Stage2_CrackTable.csv",
            mime="text/csv",
            key=f"stage2_csv_{key_prefix}",
            use_container_width=True,
        )
    pdf_buf = export_stage2_pdf(component_crack_data)
    with col_btn2:
        st.download_button(
            "📄 Download Stage 2 Knowledge Report (PDF)",
            data=pdf_buf.getvalue(),
            file_name="BKAI_Stage2_Report.pdf",
            mime="application/pdf",
            key=f"stage2_pdf_{key_prefix}",
            use_container_width=True,
        )


# =========================================================
# 6. USER STATS
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
# 7. MAIN APP
# =========================================================

def show_top_banner(username=""):
    st.markdown(
        f"""
        <div class="bkai-main-header">
            <div class="bkai-main-title">BKAI - AI-Based Concrete Crack Detection and Classification System</div>
            <div class="bkai-main-subtitle">
                {"Welcome back, " + username + ". " if username else ""}
                Upload concrete images for AI-based crack detection, segmentation, severity estimation, geometry measurement, and PDF reporting in one integrated workspace.
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
                    "Student", "Graduate Student / Researcher", "Structural Engineer", "Site Engineer",
                    "Supervision Consultant", "Construction Contractor", "Project Owner / Project Management",
                    "IT Engineer", "Lecturer / Academic Staff", "Other",
                ],
            )
        with col2:
            country = st.text_input("Country / Region", placeholder="Enter your country or region")
            project_name = st.text_input("Project / Case Name", placeholder="Optional project name")
            purpose = st.selectbox(
                "Purpose of Use",
                [
                    "Academic Research", "Thesis / Dissertation", "Site Inspection",
                    "Structural Monitoring", "Quality Control", "Training / Demonstration", "Other",
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("Measurement Settings")
    use_scale = st.sidebar.checkbox("Use scale calibration (mm/pixel)", value=False)
    mm_per_pixel = 1.0
    if use_scale:
        mm_per_pixel = st.sidebar.number_input(
            "mm per pixel",
            min_value=0.0001,
            value=0.1000,
            step=0.0001,
            format="%.4f",
        )

    with st.sidebar.expander("📊 User Statistics Manager", expanded=False):
        if user_stats:
            df_stats = pd.DataFrame(user_stats)
            st.dataframe(df_stats, use_container_width=True, height=220)
            stats_csv = df_stats.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇ Download User Statistics (CSV)", data=stats_csv, file_name="BKAI_UserStats.csv", mime="text/csv")
        else:
            st.info("No user statistics available yet.")

    st.markdown("<div class='bkai-card'>", unsafe_allow_html=True)
    st.subheader("Image Upload and Analysis")
    st.caption("Upload one or multiple concrete images. The system will perform crack detection, segmentation, geometry measurement, and report generation.")
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
            measurement_visual_img = None

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
                        orig_img, preds_conf, image_key=uploaded_file.name, min_conf=min_conf
                    )
                    st.image(analyzed_img, use_container_width=True)
                    st.markdown(
                        "<div class='bkai-status-danger'>⚠️ Conclusion: Cracks were detected in this image.</div>",
                        unsafe_allow_html=True,
                    )

            st.write("---")
            tab_stage1, tab_stage2 = st.tabs(["Stage 1 – Detailed Analysis Report", "Stage 2 – Crack Classification"])

            with tab_stage1:
                st.subheader("Crack Information Dashboard")

                confs = [float(p.get("confidence", 0)) for p in preds_conf]
                avg_conf = (sum(confs) / len(confs)) if confs else 0.0

                crack_ratio_percent, crack_area_px2 = crack_area_ratio_percent(preds_conf, img_w, img_h)
                severity = estimate_severity_from_ratio(crack_ratio_percent)

                mask_union = build_union_mask_from_predictions(preds_conf, img_w, img_h)
                measure_info = measure_crack_geometry_from_mask(mask_union)

                length_px = measure_info["length_px"]
                avg_width_px = measure_info["avg_width_px"]
                max_width_px = measure_info["max_width_px"]

                if use_scale:
                    length_value = length_px * mm_per_pixel
                    avg_width_value = avg_width_px * mm_per_pixel
                    max_width_value = max_width_px * mm_per_pixel
                    unit_text = "mm"
                else:
                    length_value = length_px
                    avg_width_value = avg_width_px
                    max_width_value = max_width_px
                    unit_text = "px"

                summary_text = (
                    "Detected cracks may indicate structural concern and should be further inspected."
                    if severity == "Severe"
                    else "Detected cracks are minor or moderate; continuous monitoring is recommended."
                )

                measurement_visual_img = create_measurement_visualization(
                    analyzed_pil_image=analyzed_img,
                    predictions=preds_conf,
                    length_value=length_value,
                    avg_width_value=avg_width_value,
                    max_width_value=max_width_value,
                    unit_text=unit_text,
                )

                st.subheader("Annotated Measurement View")
                st.image(measurement_visual_img, use_container_width=True)

                metrics = [
                    {"Metric": "Image Name", "Value": uploaded_file.name, "Description": "Uploaded image filename"},
                    {"Metric": "Total Processing Time", "Value": f"{total_time:.2f} s", "Description": "Total execution time"},
                    {"Metric": "Inference Speed", "Value": f"{total_time:.2f} s/image", "Description": "Processing time per image"},
                    {"Metric": "Average Confidence", "Value": f"{avg_conf:.2f}", "Description": "Average confidence score"},
                    {"Metric": "Crack Area (px²)", "Value": f"{crack_area_px2:.0f}", "Description": "Mask area in pixels"},
                    {"Metric": "Crack Area Ratio (%)", "Value": f"{crack_ratio_percent:.2f} %", "Description": "Crack mask area ratio"},
                    {"Metric": f"Crack Length ({unit_text})", "Value": f"{length_value:.2f}", "Description": "Estimated crack centerline length"},
                    {"Metric": f"Average Width ({unit_text})", "Value": f"{avg_width_value:.2f}", "Description": "Average crack width estimated from distance transform"},
                    {"Metric": f"Maximum Width ({unit_text})", "Value": f"{max_width_value:.2f}", "Description": "Maximum local crack width"},
                    {"Metric": "Severity Level", "Value": severity, "Description": "Severity estimated by crack ratio"},
                    {"Metric": "Timestamp", "Value": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Description": "Execution timestamp"},
                    {"Metric": "Summary", "Value": summary_text, "Description": "Automatic system conclusion"},
                ]
                metrics_df = pd.DataFrame(metrics)
                render_metrics_dashboard(metrics_df)

                with st.expander("View Metrics Table", expanded=False):
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
                    measurement_visual_img=measurement_visual_img,
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
# 8. USERS
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
    st.markdown('<div class="app-shell">', unsafe_allow_html=True)

    logo_html = '<div class="hero-logo"><div class="hero-logo-fallback">BKAI</div></div>'
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, 'rb') as f:
                logo_b64 = base64.b64encode(f.read()).decode('utf-8')
            logo_html = f'<div class="hero-logo"><img src="data:image/png;base64,{logo_b64}" alt="BKAI Logo" style="width:92px;height:auto;display:block;" /></div>'
        except Exception:
            pass

    hero_html = f"""
    <div class="hero">
        <div class="hero-flex">
            {logo_html}
            <div class="hero-copy">
                <div class="hero-kicker">BKAI CRACK ANALYSIS PORTAL</div>
                <h1 class="hero-title">AI-Based Concrete Crack Detection Platform</h1>
                <div class="hero-subtitle">
                    Secure access to image-based crack detection, segmentation, reporting, and structural crack classification in one integrated interface.
                </div>
                <div class="hero-badge">Welcome to the system</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
    st.markdown('<div class="login-card-wrap"><div class="login-card"><div class="login-card-inner">', unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        st.markdown('<div class="section-title">Sign in</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Access your BKAI workspace with your registered credentials.</div>', unsafe_allow_html=True)

        login_user = st.text_input("Username", key="login_user", placeholder="Enter username")
        login_pass = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
        st.checkbox("Stay logged in", key="stay_logged_in")

        login_btn = st.button("Log in with Credentials", key="login_button")
        st.markdown('<div class="form-sep">or</div>', unsafe_allow_html=True)
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        badge_btn = st.button("Log in with Badge", key="badge_button")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="help-note">Forgot your registered account or password? Contact the BKAI analysis portal team.</div>', unsafe_allow_html=True)

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

    with tab_register:
        st.markdown('<div class="section-title">Create account</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Register a new BKAI account to access the crack analysis workspace.</div>', unsafe_allow_html=True)

        reg_user = st.text_input("Username", key="reg_user", placeholder="Choose a username")
        reg_email = st.text_input("Email", key="reg_email", placeholder="Enter your email")
        reg_pass = st.text_input("Password", type="password", key="reg_pass", placeholder="Create a password")
        reg_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2", placeholder="Re-enter password")
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

    st.markdown('</div></div></div>', unsafe_allow_html=True)


# =========================================================
# 9. ENTRY
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
