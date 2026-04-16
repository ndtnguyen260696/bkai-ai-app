
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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
        pdfmetrics.registerFont(TTFont(FONT_NAME, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
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
        div[data-testid="stTextInput"] input{
            min-height:52px !important;
            border-radius:14px !important;
            border:1.5px solid #d8e1ef !important;
            background:#f8fbff !important;
            box-shadow:none !important;
            padding:0 16px !important;
            color:#1f2937 !important;
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
        .form-sep::before, .form-sep::after{
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
        .stage2-component{
            border:1px solid #dbe7f5;
            border-radius:18px;
            overflow:hidden;
            margin-bottom:16px;
            background:#fff;
            box-shadow:0 10px 24px rgba(31,58,120,.06);
        }
        .stage2-header{
            background:linear-gradient(135deg, #4f8cff 0%, #2b6ee9 60%, #1d5ddb 100%);
            color:#fff;
            padding:10px 16px;
            font-weight:800;
        }
        @media (max-width: 900px){
            .hero{ padding:26px 22px 96px; }
            .hero-flex{ flex-direction:column; align-items:flex-start; }
            .hero-title{ font-size:40px; }
            .hero-subtitle{ font-size:16px; }
            .hero-badge{ font-size:18px; }
            .login-card-wrap{ padding:0 14px 20px; margin-top:-64px; }
            .login-card-inner{ padding:18px 18px 22px; }
            .block-container{ max-width: 98%; }
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

def build_union_mask_from_predictions(predictions, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for p in predictions:
        pts_raw = p.get("points", None)
        if pts_raw is None:
            continue
        polys = extract_polygons(pts_raw, img_w, img_h)
        for poly in polys:
            pts = np.array(poly, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)
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
            "length_line": ((0, 0), (0, 0)),
        }

    skeleton = morphological_skeleton(mask)
    length_px = float(cv2.countNonZero(skeleton))
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    skeleton_values = dist_map[skeleton > 0]
    avg_width_px = float(np.mean(skeleton_values) * 2.0) if len(skeleton_values) > 0 else 0.0

    _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
    max_width_px = float(max_val * 2.0)

    ys, xs = np.where(skeleton > 0)
    if len(xs) > 1:
        left_idx = np.argmin(xs)
        right_idx = np.argmax(xs)
        pt1 = (int(xs[left_idx]), int(ys[left_idx]))
        pt2 = (int(xs[right_idx]), int(ys[right_idx]))
    else:
        pt1 = (0, 0)
        pt2 = (0, 0)

    return {
        "length_px": length_px,
        "avg_width_px": avg_width_px,
        "max_width_px": max_width_px,
        "skeleton": skeleton,
        "dist_map": dist_map,
        "max_point": max_loc,
        "length_line": (pt1, pt2),
    }

# =========================================================
# 1.2 DRAW RESULT
# =========================================================

def draw_predictions_with_mask(image: Image.Image, predictions, image_key: str = "", min_conf: float = 0.0):
    base = image.convert("RGB")
    W, H = base.size

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

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

        draw.rectangle([x0, max(0, y0 - 28), x0 + 130, max(0, y0 - 2)], fill=(r, g, b, 220))
        draw.text((x0 + 8, max(0, y0 - 25)), label, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(base.convert("RGBA"), overlay)
    return result.convert("RGB")

def create_measurement_visualization(
    analyzed_pil_image,
    predictions,
    length_value,
    avg_width_value,
    max_width_value,
    unit_text="px",
    panel_ratio=0.34,
):
    img_rgb = np.array(analyzed_pil_image.convert("RGB"))
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    left_w = int(w * (1.0 - panel_ratio))
    right_w = w - left_w

    crack_view = img.copy()

    center_color = (30, 210, 255)
    width_line_color = (95, 105, 255)
    point_color = (255, 255, 255)

    union_mask = np.zeros((h, w), dtype=np.uint8)
    for p in predictions:
        pts_raw = p.get("points", None)
        if pts_raw is None:
            continue
        polys = extract_polygons(pts_raw, w, h)
        for poly in polys:
            pts = np.array(poly, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(union_mask, [pts], 255)

    if cv2.countNonZero(union_mask) > 0:
        measure_info = measure_crack_geometry_from_mask(union_mask)
        skeleton = measure_info["skeleton"]
        dist_map = measure_info["dist_map"]
        max_loc = measure_info["max_point"]
        pt1, pt2 = measure_info["length_line"]

        crack_view[skeleton > 0] = center_color

        if pt1 != pt2:
            cv2.line(crack_view, pt1, pt2, center_color, 2, cv2.LINE_AA)

        _, max_val, _, _ = cv2.minMaxLoc(dist_map)
        cx, cy = int(max_loc[0]), int(max_loc[1])
        radius = int(max_val)
        x0 = max(0, cx - radius)
        x1 = min(w - 1, cx + radius)
        cv2.line(crack_view, (x0, cy), (x1, cy), width_line_color, 3, cv2.LINE_AA)
        cv2.circle(crack_view, (cx, cy), 5, point_color, -1)
        cv2.circle(crack_view, (cx, cy), 8, width_line_color, 2)

    crack_view = cv2.resize(crack_view, (left_w, h))

    panel = np.zeros((h, right_w, 3), dtype=np.uint8)
    panel[:] = (58, 86, 132)

    title_fs = 0.52
    value_fs = 0.84
    title_th = 1
    value_th = 2

    rows_y = [int(h * 0.22), int(h * 0.50), int(h * 0.78)]
    labels = ["Length", "Avg Width", "Max Width"]
    values = [
        f"{length_value:.1f} {unit_text}",
        f"{avg_width_value:.2f} {unit_text}",
        f"{max_width_value:.2f} {unit_text}",
    ]
    row_colors = [(255, 170, 80), (120, 220, 120), (110, 120, 255)]

    for i, ymid in enumerate(rows_y):
        c = row_colors[i]
        cv2.line(panel, (20, ymid - 8), (50, ymid - 8), c, 2)
        cv2.circle(panel, (20, ymid - 8), 4, c, -1)
        cv2.circle(panel, (50, ymid - 8), 4, c, -1)
        cv2.putText(panel, labels[i], (68, ymid - 8), cv2.FONT_HERSHEY_SIMPLEX, title_fs, (255, 255, 255), title_th, cv2.LINE_AA)
        cv2.putText(panel, values[i], (68, ymid + 22), cv2.FONT_HERSHEY_SIMPLEX, value_fs, c, value_th, cv2.LINE_AA)
        if i < 2:
            cv2.line(panel, (16, ymid + 44), (right_w - 16, ymid + 44), (105, 124, 156), 1)

    cv2.rectangle(panel, (0, 0), (right_w - 1, h - 1), (72, 98, 145), 2)

    final = np.concatenate([crack_view, panel], axis=1)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final)

# =========================================================
# 1.3 SEVERITY / METRICS
# =========================================================

def estimate_severity_from_ratio(area_ratio_percent: float):
    r = float(area_ratio_percent)
    if r < 0.2:
        return "Minor"
    elif r < 1.0:
        return "Moderate"
    return "Severe"

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
        row_items = normal_rows[i:i+cols_per_row]
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
# 2. PDF EXPORT
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
    BODY_FONT = FONT_NAME

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
        c.setFont(TITLE_FONT, 18)
        c.drawCentredString(page_w / 2.0, y_top - 6 * mm, page_title)

        if subtitle:
            c.setFont(BODY_FONT, 11)
            c.drawCentredString(page_w / 2.0, y_top - 13 * mm, subtitle)

        footer_y = BOTTOM - 6
        c.setFont(BODY_FONT, 8)
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

    # page 1
    content_top_y = draw_header("ANALYSIS REPORT", page_no=1)
    content_top_y -= 5 * mm
    gap_x = 10 * mm
    slot_w = (CONTENT_W - gap_x) / 2.0
    max_img_h = 88 * mm

    c.setFont(BODY_FONT, 11)
    c.setFillColor(colors.black)
    c.drawString(LEFT, content_top_y + 4 * mm, "Original Image")
    c.drawString(LEFT + slot_w + gap_x, content_top_y + 4 * mm, "Analyzed Image")

    left_bottom = draw_pil_image(original_img, LEFT, content_top_y, slot_w, max_img_h)
    right_bottom = draw_pil_image(analyzed_img, LEFT + slot_w + gap_x, content_top_y, slot_w, max_img_h)
    images_bottom_y = min(left_bottom, right_bottom)

    # charts
    charts_top_y = images_bottom_y - 20 * mm
    max_chart_h = 62 * mm

    if chart_bar_png is not None:
        chart_bar_png.seek(0)
        bar_img = ImageReader(chart_bar_png)
        bw, bh = bar_img.getSize()
        scale_bar = min(slot_w / bw, max_chart_h / bh)
        cw = bw * scale_bar
        ch = bh * scale_bar
        bar_bottom = charts_top_y - ch
        c.drawImage(bar_img, LEFT, bar_bottom, width=cw, height=ch, mask="auto")
        c.setFont(BODY_FONT, 10)
        c.drawString(LEFT, bar_bottom - 10, "Confidence of each detected crack region")

    if chart_pie_png is not None:
        chart_pie_png.seek(0)
        pie_img = ImageReader(chart_pie_png)
        pw, ph = pie_img.getSize()
        scale_pie = min(slot_w / pw, max_chart_h / ph)
        cw = pw * scale_pie
        ch = ph * scale_pie
        pie_bottom = charts_top_y - ch
        c.drawImage(pie_img, LEFT + slot_w + gap_x, pie_bottom, width=cw, height=ch, mask="auto")
        c.setFont(BODY_FONT, 10)
        c.drawString(LEFT + slot_w + gap_x, pie_bottom - 10, "Crack region ratio relative to full image")

    c.showPage()

    # page 2
    content_top_y = draw_header("ANALYSIS REPORT", subtitle="Annotated Measurement View and Metrics", page_no=2)

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
        content_top_y = y_bottom - 10 * mm

    rows = []
    if metrics_df is not None:
        for _, r in metrics_df.iterrows():
            rows.append((str(r.get("Metric", "")), str(r.get("Value", ""))))

    col1_w = 70 * mm
    col2_w = CONTENT_W - col1_w
    row_h = 9 * mm

    c.setFillColor(colors.HexColor("#1e88e5"))
    c.rect(LEFT, content_top_y - row_h, CONTENT_W, row_h, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont(BODY_FONT, 10)
    c.drawString(LEFT + 3, content_top_y - 6 * mm, "Metric")
    c.drawString(LEFT + col1_w + 3, content_top_y - 6 * mm, "Value")

    y = content_top_y - row_h
    for i, (label, val) in enumerate(rows):
        if y - row_h < BOTTOM + 10 * mm:
            c.showPage()
            y = draw_header("ANALYSIS REPORT", subtitle="Metrics (continued)", page_no=3)
            c.setFillColor(colors.HexColor("#1e88e5"))
            c.rect(LEFT, y - row_h, CONTENT_W, row_h, stroke=0, fill=1)
            c.setFillColor(colors.white)
            c.setFont(BODY_FONT, 10)
            c.drawString(LEFT + 3, y - 6 * mm, "Metric")
            c.drawString(LEFT + col1_w + 3, y - 6 * mm, "Value")
            y -= row_h

        if i % 2 == 0:
            c.setFillColor(colors.HexColor("#f7fbff"))
            c.rect(LEFT, y - row_h, CONTENT_W, row_h, stroke=0, fill=1)

        c.setStrokeColor(colors.HexColor("#dbe7f5"))
        c.rect(LEFT, y - row_h, CONTENT_W, row_h, stroke=1, fill=0)
        c.line(LEFT + col1_w, y, LEFT + col1_w, y - row_h)

        c.setFillColor(colors.black)
        c.setFont(BODY_FONT, 9)
        c.drawString(LEFT + 3, y - 5.7 * mm, label[:48])
        c.drawString(LEFT + col1_w + 3, y - 5.7 * mm, val[:60])

        y -= row_h

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

    y_top = page_h - TOP
    if os.path.exists(LOGO_PATH):
        try:
            logo = ImageReader(LOGO_PATH)
            logo_w = 30 * mm
            iw, ih = logo.getSize()
            logo_h = logo_w * ih / iw
            c.drawImage(logo, LEFT, y_top - logo_h, width=logo_w, height=logo_h, mask="auto")
        except Exception:
            pass

    c.setFont(FONT_NAME, 18)
    c.drawCentredString(page_w / 2, y_top - 6 * mm, "ANALYSIS REPORT")
    c.setFont(FONT_NAME, 11)
    c.drawCentredString(page_w / 2, y_top - 14 * mm, "Case: No significant crack detected")

    content_top_y = y_top - 40 * mm
    slot_w = CONTENT_W
    max_img_h = 90 * mm
    img = ImageReader(original_img)
    iw, ih = img.getSize()
    scale = min(slot_w / iw, max_img_h / ih, 1.0)
    w = iw * scale
    h = ih * scale
    c.drawImage(img, LEFT + (CONTENT_W - w) / 2, content_top_y - h, width=w, height=h, mask="auto")

    banner_y = content_top_y - h - 16 * mm
    c.setFillColor(colors.HexColor("#e8f5e9"))
    c.rect(LEFT, banner_y, CONTENT_W, 16 * mm, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#2e7d32"))
    c.drawString(LEFT + 4 * mm, banner_y + 8 * mm, "No clearly visible cracks were detected in the image under the current model threshold.")

    footer_y = BOTTOM - 6
    c.setFont(FONT_NAME, 8)
    c.setFillColor(colors.grey)
    c.drawString(LEFT, footer_y, f"BKAI – Concrete Crack Inspection | Generated at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    c.drawRightString(page_w - RIGHT, footer_y, "Page 1")
    c.save()
    buf.seek(0)
    return buf

# =========================================================
# 3. STAGE 2
# =========================================================

def render_component_crack_table(component_df: pd.DataFrame):
    if component_df is None or component_df.empty:
        st.info("No Stage 2 crack classification data available.")
        return

    for component, subdf in component_df.groupby("Component", sort=False):
        st.markdown(f'<div class="stage2-component"><div class="stage2-header">{component}</div>', unsafe_allow_html=True)
        display_df = subdf[["Crack Type", "Cause", "Shape Characteristics"]].reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

def show_stage2_demo(key_prefix="stage2"):
    st.subheader("Stage 2 – Crack Classification and Suggested Causes / Actions")

    component_crack_data = pd.DataFrame([
        {"Component":"Beam","Crack Type":"Flexural Crack","Cause":"Bending moment exceeds the allowable limit; inadequate flexural reinforcement or section capacity.","Shape Characteristics":"Usually appears at mid-span and is widest in the tension zone."},
        {"Component":"Beam","Crack Type":"Shear Crack","Cause":"High shear force; inadequate concrete shear capacity or insufficient stirrups.","Shape Characteristics":"Inclined crack, often around 45° relative to the beam axis."},
        {"Component":"Beam","Crack Type":"Torsional Crack","Cause":"Insufficient torsional reinforcement or unsuitable cross-section design.","Shape Characteristics":"Diagonal or zigzag pattern around the beam surface."},
        {"Component":"Beam","Crack Type":"Corrosion-Induced Crack","Cause":"Aggressive environment, thin cover depth, and steel corrosion expansion.","Shape Characteristics":"Runs along reinforcement lines and may be accompanied by rust staining or cover spalling."},
        {"Component":"Column","Crack Type":"Diagonal Crack","Cause":"High combined compression, bending, or shear.","Shape Characteristics":"Inclined cracks appear on the surface when the load approaches or exceeds capacity."},
        {"Component":"Column","Crack Type":"Splitting / Longitudinal Crack","Cause":"Longitudinal splitting due to high compressive stress or weak concrete.","Shape Characteristics":"Multiple parallel vertical cracks."},
        {"Component":"Slab","Crack Type":"Plastic Shrinkage Crack","Cause":"Rapid moisture evaporation while concrete is still plastic.","Shape Characteristics":"Shallow and small cracks, often forming a polygonal pattern."},
        {"Component":"Slab","Crack Type":"Drying Shrinkage Crack","Cause":"Shrinkage after hardening in dry or hot environments.","Shape Characteristics":"Map cracking or relatively straight crack lines."},
        {"Component":"Concrete Wall","Crack Type":"Shrinkage Crack","Cause":"Rapid moisture loss; shrinkage stress exceeds tensile capacity.","Shape Characteristics":"Random, polygonal, or intersecting crack pattern."},
        {"Component":"Concrete Wall","Crack Type":"Thermal Crack","Cause":"Temperature difference through the wall thickness.","Shape Characteristics":"Often vertical and wider in the thermal tension zone."},
    ])

    render_component_crack_table(component_crack_data)
    st.caption("Component-based crack knowledge reference.")

# =========================================================
# 4. USER DATA
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
if "profile_filled" not in st.session_state:
    st.session_state.profile_filled = False

# =========================================================
# 5. MAIN UI
# =========================================================

def show_top_banner(username=""):
    st.markdown(
        f"""
        <div class="bkai-main-header">
            <div class="bkai-main-title">BKAI - AI-Based Concrete Crack Detection and Classification System</div>
            <div class="bkai-main-subtitle">
                {"Welcome back, " + username + ". " if username else ""}
                Upload concrete images for AI-based crack detection, segmentation, measurement, and PDF reporting in one integrated workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_profile_form():
    if st.session_state.profile_filled:
        return True

    st.markdown("<div class='bkai-card'>", unsafe_allow_html=True)
    st.subheader("User Profile Information")
    st.caption("Please complete the information below before starting the analysis.")

    with st.form("user_info_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name *")
            email = st.text_input("Email *")
            organization = st.text_input("Organization / Company")
            occupation = st.selectbox(
                "Occupation / User Group *",
                ["Student","Graduate Student / Researcher","Structural Engineer","Site Engineer","Consultant","Contractor","Owner / PMU","Lecturer / Academic Staff","Other"],
            )
        with col2:
            country = st.text_input("Country / Region")
            project_name = st.text_input("Project / Case Name")
            purpose = st.selectbox("Purpose of Use", ["Academic Research","Thesis / Dissertation","Site Inspection","Structural Monitoring","Quality Control","Training / Demonstration","Other"])
            notes = st.text_area("Remarks / Notes", height=110)

        submit_info = st.form_submit_button("Save Profile and Start Analysis")

    if submit_info:
        if not full_name or not occupation or not email:
            st.warning("Please complete all required fields: Full Name, Occupation, and Email.")
            st.markdown("</div>", unsafe_allow_html=True)
            return False
        if "@" not in email or "." not in email:
            st.warning("Invalid email address. Please check and try again.")
            st.markdown("</div>", unsafe_allow_html=True)
            return False

        st.session_state.profile_filled = True
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
    st.sidebar.subheader("Measurement Settings")
    use_scale = st.sidebar.checkbox("Use scale calibration (mm/pixel)", value=False)
    mm_per_pixel = 1.0
    if use_scale:
        mm_per_pixel = st.sidebar.number_input("mm per pixel", min_value=0.0001, value=0.1000, step=0.0001, format="%.4f")

    st.markdown("<div class='bkai-card'>", unsafe_allow_html=True)
    st.subheader("Image Upload and Analysis")
    uploaded_files = st.file_uploader("Upload one or more concrete images (JPG / PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    analyze_btn = st.button("Analyze Images")
    st.markdown("</div>", unsafe_allow_html=True)

    if not analyze_btn:
        return

    if not uploaded_files:
        st.warning("Please upload at least one image before clicking Analyze Images.")
        return

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

        with col2:
            st.subheader("Analyzed Image")
            if len(preds_conf) == 0:
                st.image(orig_img, use_container_width=True)
                st.markdown("<div class='bkai-status-ok'>✅ Conclusion: No clearly visible crack was detected in this image.</div>", unsafe_allow_html=True)
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
                analyzed_img = draw_predictions_with_mask(orig_img, preds_conf, image_key=uploaded_file.name, min_conf=min_conf)
                st.image(analyzed_img, use_container_width=True)
                st.markdown("<div class='bkai-status-danger'>⚠️ Conclusion: Cracks were detected in this image.</div>", unsafe_allow_html=True)

        union_mask = build_union_mask_from_predictions(preds_conf, img_w, img_h)
        measure_info = measure_crack_geometry_from_mask(union_mask)
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

        measurement_visual_img = create_measurement_visualization(
            analyzed_pil_image=analyzed_img,
            predictions=preds_conf,
            length_value=length_value,
            avg_width_value=avg_width_value,
            max_width_value=max_width_value,
            unit_text=unit_text,
        )

        st.write("---")
        tab_stage1, tab_stage2 = st.tabs(["Stage 1 – Detailed Analysis Report", "Stage 2 – Crack Classification"])

        with tab_stage1:
            st.subheader("Annotated Measurement View")
            st.image(measurement_visual_img, use_container_width=True)

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
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

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
# 6. AUTH PAGE
# =========================================================

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
# 7. ENTRY
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
