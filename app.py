import base64
import datetime as dt
import hashlib
import io
import json
import math
import os
import secrets
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from skimage.morphology import skeletonize


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "BKAI"
APP_TITLE = "AI-Based Concrete Crack Inspection Platform"
APP_VERSION = "2.0"
MODEL_DEFAULT = "crack_segmentation_detection/4"

DB_PATH = "bkai.db"
LOGO_PATH = "BKAI_Logo.png"

MAX_FILE_SIZE_MB = 12
MAX_IMAGE_DIMENSION = 4096
REQUEST_TIMEOUT_SECONDS = 90

st.set_page_config(
    page_title=f"{APP_NAME} – Concrete Crack Inspection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. GLOBAL USER INTERFACE STYLE
# ============================================================

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root{
            --primary:#2563eb;
            --primary-dark:#174bb8;
            --primary-soft:#eaf2ff;
            --surface:#ffffff;
            --surface-soft:#f7faff;
            --text:#0f172a;
            --muted:#64748b;
            --border:#dce6f2;
            --success:#15803d;
            --warning:#b45309;
            --danger:#b91c1c;
        }

        html, body, [class*="css"]{
            font-family: Inter, "Segoe UI", Arial, sans-serif;
            color:var(--text);
        }

        .stApp{
            background:
                radial-gradient(circle at 0% 0%, rgba(255,255,255,.98), rgba(232,240,252,.82) 25%, transparent 48%),
                linear-gradient(180deg,#edf3fb 0%,#dfe8f5 100%);
        }

        .block-container{
            max-width:1260px;
            padding-top:1rem;
            padding-bottom:3rem;
        }

        [data-testid="stSidebar"]{
            background:linear-gradient(180deg,#fbfdff 0%,#eef4fc 100%);
            border-right:1px solid #d9e4f0;
        }

        [data-testid="stSidebar"] .block-container{
            padding-top:1rem;
        }

        .hero{
            position:relative;
            overflow:hidden;
            background:
                radial-gradient(circle at 90% 10%, rgba(255,255,255,.22), transparent 28%),
                linear-gradient(135deg,#4f8cff 0%,#2563eb 56%,#174bb8 100%);
            color:#fff;
            border-radius:28px;
            padding:28px 30px;
            margin:4px 0 20px;
            box-shadow:0 24px 60px rgba(35,72,150,.20);
        }

        .hero:after{
            content:"";
            position:absolute;
            right:-80px;
            bottom:-110px;
            width:280px;
            height:280px;
            border-radius:50%;
            border:42px solid rgba(255,255,255,.08);
        }

        .hero-kicker{
            font-size:12px;
            letter-spacing:.20em;
            text-transform:uppercase;
            font-weight:800;
            opacity:.88;
            margin-bottom:8px;
        }

        .hero-title{
            font-size:36px;
            line-height:1.15;
            font-weight:850;
            margin:0 0 10px;
            max-width:900px;
        }

        .hero-subtitle{
            font-size:16px;
            line-height:1.75;
            opacity:.94;
            max-width:920px;
        }

        .hero-badges{
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            margin-top:18px;
        }

        .hero-badge{
            padding:8px 12px;
            border-radius:999px;
            border:1px solid rgba(255,255,255,.24);
            background:rgba(255,255,255,.10);
            font-size:12px;
            font-weight:700;
        }

        .panel{
            background:linear-gradient(180deg,rgba(255,255,255,.98) 0%,rgba(248,251,255,.98) 100%);
            border:1px solid var(--border);
            border-radius:22px;
            padding:20px;
            margin-bottom:16px;
            box-shadow:0 14px 32px rgba(31,58,120,.08);
        }

        .panel-title{
            font-size:18px;
            font-weight:800;
            color:var(--text);
            margin-bottom:4px;
        }

        .panel-subtitle{
            font-size:13px;
            color:var(--muted);
            line-height:1.6;
            margin-bottom:14px;
        }

        .metric-card{
            min-height:118px;
            background:linear-gradient(180deg,#ffffff 0%,#f7faff 100%);
            border:1px solid var(--border);
            border-radius:18px;
            padding:15px 16px;
            margin-bottom:12px;
            box-shadow:0 10px 24px rgba(31,58,120,.07);
        }

        .metric-label{
            font-size:11px;
            color:var(--muted);
            text-transform:uppercase;
            letter-spacing:.08em;
            font-weight:800;
            margin-bottom:9px;
        }

        .metric-value{
            font-size:24px;
            font-weight:850;
            color:var(--text);
            line-height:1.15;
            word-break:break-word;
        }

        .metric-note{
            font-size:12px;
            color:var(--muted);
            line-height:1.45;
            margin-top:7px;
        }

        .status{
            border-radius:15px;
            padding:12px 14px;
            font-weight:750;
            line-height:1.55;
            margin:10px 0;
        }

        .status-success{
            background:#ecfdf3;
            color:#166534;
            border:1px solid #bbf7d0;
        }

        .status-warning{
            background:#fff7ed;
            color:#9a3412;
            border:1px solid #fed7aa;
        }

        .status-danger{
            background:#fef2f2;
            color:#991b1b;
            border:1px solid #fecaca;
        }

        .status-info{
            background:#eff6ff;
            color:#1d4ed8;
            border:1px solid #bfdbfe;
        }

        .sidebar-user{
            background:linear-gradient(135deg,#eef5ff,#e2edff);
            border:1px solid #c9dcfa;
            color:#1d4ed8;
            border-radius:14px;
            padding:12px 13px;
            font-weight:800;
            margin-bottom:12px;
        }

        .inspection-chip{
            display:inline-block;
            padding:7px 11px;
            margin:0 5px 5px 0;
            border-radius:999px;
            background:#edf4ff;
            border:1px solid #cfe0fb;
            color:#1d4ed8;
            font-size:12px;
            font-weight:750;
        }

        .section-heading{
            font-size:22px;
            font-weight:850;
            color:#10213f;
            margin:12px 0 10px;
        }

        .small-muted{
            color:var(--muted);
            font-size:12px;
            line-height:1.55;
        }

        div[data-testid="stDataFrame"]{
            border:1px solid var(--border);
            border-radius:16px;
            overflow:hidden;
        }

        .stButton > button, .stDownloadButton > button{
            border-radius:12px;
            font-weight:800;
            min-height:42px;
        }

        .stTabs [data-baseweb="tab-list"]{
            gap:7px;
        }

        .stTabs [data-baseweb="tab"]{
            border-radius:12px 12px 0 0;
            padding:10px 14px;
            font-weight:750;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()


# ============================================================
# 3. SECRET CONFIGURATION
# ============================================================

def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


ROBOFLOW_API_KEY = read_secret("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL = read_secret("ROBOFLOW_MODEL", MODEL_DEFAULT)


# ============================================================
# 4. DATABASE AND AUTHENTICATION
# ============================================================

def db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL,
                occupation TEXT NOT NULL,
                organization TEXT,
                country TEXT,
                project_name TEXT,
                purpose TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def create_password_hash(
    password: str,
    salt_hex: Optional[str] = None,
) -> Tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        220_000,
    )
    return digest.hex(), salt.hex()


def register_account(
    username: str,
    email: str,
    password: str,
) -> Tuple[bool, str]:
    password_hash, salt = create_password_hash(password)

    try:
        with db_connection() as conn:
            conn.execute(
                """
                INSERT INTO users(
                    username,email,password_hash,salt,created_at
                ) VALUES(?,?,?,?,?)
                """,
                (
                    username.strip(),
                    email.strip(),
                    password_hash,
                    salt,
                    dt.datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "This username already exists."


def authenticate(username: str, password: str) -> bool:
    with db_connection() as conn:
        account = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username.strip(),),
        ).fetchone()

    if account is None:
        return False

    candidate_hash, _ = create_password_hash(
        password,
        account["salt"],
    )

    return secrets.compare_digest(
        candidate_hash,
        account["password_hash"],
    )


def save_user_profile(profile: Dict[str, str]) -> None:
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO profiles(
                username,full_name,email,occupation,
                organization,country,project_name,
                purpose,notes,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                profile["username"],
                profile["full_name"],
                profile["email"],
                profile["occupation"],
                profile.get("organization", ""),
                profile.get("country", ""),
                profile.get("project_name", ""),
                profile.get("purpose", ""),
                profile.get("notes", ""),
                dt.datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


# ============================================================
# 5. DATA OBJECTS
# ============================================================

@dataclass
class CrackResult:
    crack_id: str
    confidence: float
    area_px2: float
    area_ratio_percent: float
    length_px: float
    avg_width_px: float
    max_width_px: float
    orientation_deg: float
    tortuosity: float
    length_value: float
    avg_width_value: float
    max_width_value: float
    area_value: float
    length_unit: str
    area_unit: str


# ============================================================
# 6. INPUT VALIDATION AND POLYGON PARSING
# ============================================================

def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_uploaded_image(uploaded_file) -> Image.Image:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"File size exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )

    raw = uploaded_file.getvalue()

    try:
        validation_image = Image.open(io.BytesIO(raw))
        validation_image.verify()
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted image: {exc}") from exc

    image = Image.open(io.BytesIO(raw)).convert("RGB")

    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail(
            (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Image.Resampling.LANCZOS,
        )

    return image


def extract_polygon_points(
    points_field: Any,
    image_width: int,
    image_height: int,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []

    def append_point(x_value: Any, y_value: Any) -> None:
        try:
            points.append((float(x_value), float(y_value)))
        except (TypeError, ValueError):
            return

    if isinstance(points_field, list):
        for point in points_field:
            if isinstance(point, dict) and "x" in point and "y" in point:
                append_point(point["x"], point["y"])
            elif isinstance(point, (list, tuple)) and len(point) == 2:
                append_point(point[0], point[1])

    if len(points) < 3:
        return []

    normalized_x = max(x for x, _ in points) <= 1.5
    normalized_y = max(y for _, y in points) <= 1.5

    output: List[Tuple[float, float]] = []

    for x_value, y_value in points:
        if normalized_x:
            x_value *= image_width
        if normalized_y:
            y_value *= image_height

        output.append(
            (
                max(0.0, min(image_width - 1.0, x_value)),
                max(0.0, min(image_height - 1.0, y_value)),
            )
        )

    return output


def extract_polygons(
    points_field: Any,
    image_width: int,
    image_height: int,
) -> List[List[Tuple[float, float]]]:
    polygons: List[List[Tuple[float, float]]] = []

    if isinstance(points_field, dict):
        for key in sorted(points_field.keys(), key=str):
            polygon = extract_polygon_points(
                points_field[key],
                image_width,
                image_height,
            )
            if len(polygon) >= 3:
                polygons.append(polygon)

    elif isinstance(points_field, list):
        polygon = extract_polygon_points(
            points_field,
            image_width,
            image_height,
        )
        if len(polygon) >= 3:
            polygons.append(polygon)

    return polygons


def prediction_mask(
    prediction: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    for polygon in extract_polygons(
        prediction.get("points"),
        image_width,
        image_height,
    ):
        points = np.array(polygon, dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)

    return mask


def union_prediction_mask(
    predictions: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    union_mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    for prediction in predictions:
        union_mask = cv2.bitwise_or(
            union_mask,
            prediction_mask(
                prediction,
                image_width,
                image_height,
            ),
        )

    return union_mask


# ============================================================
# 7. CRACK GEOMETRY
# ============================================================

def clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return binary


def weighted_skeleton_length(skeleton: np.ndarray) -> float:
    """
    Horizontal/vertical connections = 1 px.
    Diagonal connections = sqrt(2) px.
    """
    skeleton_bool = skeleton > 0

    horizontal = np.sum(
        skeleton_bool[:, :-1] & skeleton_bool[:, 1:]
    )
    vertical = np.sum(
        skeleton_bool[:-1, :] & skeleton_bool[1:, :]
    )
    diagonal_a = np.sum(
        skeleton_bool[:-1, :-1] & skeleton_bool[1:, 1:]
    )
    diagonal_b = np.sum(
        skeleton_bool[:-1, 1:] & skeleton_bool[1:, :-1]
    )

    return float(
        horizontal
        + vertical
        + math.sqrt(2.0) * (diagonal_a + diagonal_b)
    )


def find_skeleton_endpoints(
    skeleton: np.ndarray,
) -> List[Tuple[int, int]]:
    skeleton_binary = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)

    neighbour_count = cv2.filter2D(
        skeleton_binary,
        cv2.CV_16S,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    ) - skeleton_binary.astype(np.int16)

    y_coordinates, x_coordinates = np.where(
        (skeleton_binary == 1) & (neighbour_count == 1)
    )

    return [
        (int(x_value), int(y_value))
        for x_value, y_value
        in zip(x_coordinates, y_coordinates)
    ]


def farthest_point_pair(
    points: List[Tuple[int, int]],
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if len(points) < 2:
        return None

    best_pair = None
    best_distance = -1.0

    for first_index in range(len(points)):
        for second_index in range(first_index + 1, len(points)):
            first_point = points[first_index]
            second_point = points[second_index]

            distance = math.hypot(
                first_point[0] - second_point[0],
                first_point[1] - second_point[1],
            )

            if distance > best_distance:
                best_distance = distance
                best_pair = (first_point, second_point)

    return best_pair


def dominant_orientation(skeleton: np.ndarray) -> float:
    y_coordinates, x_coordinates = np.where(skeleton > 0)

    if len(x_coordinates) < 2:
        return 0.0

    coordinates = np.column_stack(
        (x_coordinates, y_coordinates)
    ).astype(np.float64)

    coordinates -= coordinates.mean(axis=0, keepdims=True)

    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]

    angle = abs(
        math.degrees(
            math.atan2(
                principal_vector[1],
                principal_vector[0],
            )
        )
    )

    if angle > 90.0:
        angle = 180.0 - angle

    return float(angle)


def measure_mask_geometry(mask: np.ndarray) -> Dict[str, Any]:
    binary = clean_mask(mask)

    if binary.size == 0 or cv2.countNonZero(binary) == 0:
        empty = np.zeros_like(binary, dtype=np.uint8)
        return {
            "skeleton": empty,
            "length_px": 0.0,
            "avg_width_px": 0.0,
            "max_width_px": 0.0,
            "orientation_deg": 0.0,
            "tortuosity": 0.0,
            "endpoints": [],
            "farthest_pair": None,
            "max_width_point": (0, 0),
        }

    skeleton_bool = skeletonize(binary > 0)
    skeleton = skeleton_bool.astype(np.uint8) * 255

    length_px = weighted_skeleton_length(skeleton)

    distance_map = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5,
    )

    local_radius = distance_map[skeleton > 0]

    average_width_px = (
        float(np.mean(local_radius) * 2.0)
        if local_radius.size > 0
        else 0.0
    )
    maximum_width_px = (
        float(np.max(local_radius) * 2.0)
        if local_radius.size > 0
        else 0.0
    )

    endpoints = find_skeleton_endpoints(skeleton)
    endpoint_pair = farthest_point_pair(endpoints)

    if endpoint_pair:
        straight_distance = math.hypot(
            endpoint_pair[0][0] - endpoint_pair[1][0],
            endpoint_pair[0][1] - endpoint_pair[1][1],
        )
    else:
        straight_distance = 0.0

    tortuosity = (
        float(length_px / straight_distance)
        if straight_distance > 0
        else 1.0
    )

    _, _, _, maximum_location = cv2.minMaxLoc(distance_map)

    return {
        "skeleton": skeleton,
        "length_px": length_px,
        "avg_width_px": average_width_px,
        "max_width_px": maximum_width_px,
        "orientation_deg": dominant_orientation(skeleton),
        "tortuosity": tortuosity,
        "endpoints": endpoints,
        "farthest_pair": endpoint_pair,
        "max_width_point": maximum_location,
    }


def create_crack_result(
    prediction: Dict[str, Any],
    crack_index: int,
    image_width: int,
    image_height: int,
    use_scale: bool,
    millimetres_per_pixel: float,
) -> CrackResult:
    mask = prediction_mask(
        prediction,
        image_width,
        image_height,
    )
    geometry = measure_mask_geometry(mask)

    area_px2 = float(cv2.countNonZero(mask))
    total_image_area = float(image_width * image_height)

    area_ratio_percent = (
        area_px2 / total_image_area * 100.0
        if total_image_area > 0
        else 0.0
    )

    if use_scale:
        length_value = (
            geometry["length_px"] * millimetres_per_pixel
        )
        average_width_value = (
            geometry["avg_width_px"] * millimetres_per_pixel
        )
        maximum_width_value = (
            geometry["max_width_px"] * millimetres_per_pixel
        )
        area_value = (
            area_px2 * millimetres_per_pixel ** 2
        )
        length_unit = "mm"
        area_unit = "mm²"
    else:
        length_value = geometry["length_px"]
        average_width_value = geometry["avg_width_px"]
        maximum_width_value = geometry["max_width_px"]
        area_value = area_px2
        length_unit = "px"
        area_unit = "px²"

    return CrackResult(
        crack_id=f"C{crack_index:02d}",
        confidence=to_float(prediction.get("confidence")),
        area_px2=area_px2,
        area_ratio_percent=area_ratio_percent,
        length_px=geometry["length_px"],
        avg_width_px=geometry["avg_width_px"],
        max_width_px=geometry["max_width_px"],
        orientation_deg=geometry["orientation_deg"],
        tortuosity=geometry["tortuosity"],
        length_value=length_value,
        avg_width_value=average_width_value,
        max_width_value=maximum_width_value,
        area_value=area_value,
        length_unit=length_unit,
        area_unit=area_unit,
    )


# ============================================================
# 8. ASSESSMENT LOGIC
# ============================================================

def classify_image_extent(area_ratio_percent: float) -> str:
    """
    This describes only the crack extent inside the image.
    It is not a structural severity assessment.
    """
    if area_ratio_percent < 0.2:
        return "Low"
    if area_ratio_percent < 1.0:
        return "Moderate"
    return "High"


def research_width_grade(maximum_width_mm: float) -> str:
    """
    Research-defined demonstration thresholds.

    Replace these thresholds with the engineering standard
    explicitly adopted and justified in the manuscript.
    """
    if maximum_width_mm < 0.2:
        return "Grade I"
    if maximum_width_mm < 0.5:
        return "Grade II"
    if maximum_width_mm < 1.0:
        return "Grade III"
    return "Grade IV"


# ============================================================
# 9. REMOTE INFERENCE
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def run_inference(
    image_bytes: bytes,
    api_key: str,
    model_name: str,
) -> Dict[str, Any]:
    response = requests.post(
        f"https://detect.roboflow.com/{model_name}",
        params={"api_key": api_key},
        files={
            "file": (
                "image.jpg",
                image_bytes,
                "image/jpeg",
            )
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("The inference response is not a JSON object.")

    predictions = payload.get("predictions", [])

    if not isinstance(predictions, list):
        raise ValueError("The predictions field is invalid.")

    return payload


# ============================================================
# 10. IMAGE VISUALIZATION
# ============================================================

def draw_segmentation_overlay(
    image: Image.Image,
    predictions: List[Dict[str, Any]],
) -> Image.Image:
    base_image = image.convert("RGB")
    image_width, image_height = base_image.size

    overlay = Image.new(
        "RGBA",
        base_image.size,
        (0, 0, 0, 0),
    )
    drawing = ImageDraw.Draw(overlay)

    for index, prediction in enumerate(predictions, start=1):
        confidence = to_float(prediction.get("confidence"))
        color = (47, 174, 255, 255)

        for polygon in extract_polygons(
            prediction.get("points"),
            image_width,
            image_height,
        ):
            drawing.polygon(
                polygon,
                fill=(47, 174, 255, 82),
            )
            drawing.line(
                polygon + [polygon[0]],
                fill=color,
                width=3,
            )

        x_center = to_float(prediction.get("x"))
        y_center = to_float(prediction.get("y"))
        box_width = to_float(prediction.get("width"))
        box_height = to_float(prediction.get("height"))

        if box_width > 0 and box_height > 0:
            x_min = max(0.0, x_center - box_width / 2)
            y_min = max(0.0, y_center - box_height / 2)
            x_max = min(
                image_width - 1.0,
                x_center + box_width / 2,
            )
            y_max = min(
                image_height - 1.0,
                y_center + box_height / 2,
            )

            drawing.rectangle(
                [x_min, y_min, x_max, y_max],
                outline=color,
                width=3,
            )

            label = f"C{index:02d} | {confidence * 100:.0f}%"
            label_top = max(0.0, y_min - 30)

            drawing.rectangle(
                [
                    x_min,
                    label_top,
                    min(image_width - 1.0, x_min + 150),
                    min(image_height - 1.0, label_top + 28),
                ],
                fill=(0, 0, 0, 210),
            )
            drawing.text(
                (x_min + 8, label_top + 5),
                label,
                fill=color,
            )

    return Image.alpha_composite(
        base_image.convert("RGBA"),
        overlay,
    ).convert("RGB")


def draw_measurement_overlay(
    image: Image.Image,
    predictions: List[Dict[str, Any]],
) -> Image.Image:
    canvas = cv2.cvtColor(
        np.array(image.convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )

    image_height, image_width = canvas.shape[:2]

    for index, prediction in enumerate(predictions, start=1):
        mask = prediction_mask(
            prediction,
            image_width,
            image_height,
        )
        geometry = measure_mask_geometry(mask)

        y_coordinates, x_coordinates = np.where(
            geometry["skeleton"] > 0
        )
        canvas[y_coordinates, x_coordinates] = (0, 255, 255)

        for endpoint in geometry["endpoints"]:
            cv2.circle(
                canvas,
                endpoint,
                5,
                (0, 220, 0),
                -1,
                cv2.LINE_AA,
            )

        maximum_width_point = geometry["max_width_point"]

        cv2.circle(
            canvas,
            maximum_width_point,
            8,
            (70, 70, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"C{index:02d}",
            (
                maximum_width_point[0] + 8,
                maximum_width_point[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    )


# ============================================================
# 11. UI COMPONENTS
# ============================================================

def logo_as_base64() -> Optional[str]:
    if not os.path.exists(LOGO_PATH):
        return None

    try:
        return base64.b64encode(
            open(LOGO_PATH, "rb").read()
        ).decode("utf-8")
    except Exception:
        return None


def show_hero(username: str = "") -> None:
    welcome_text = (
        f"Welcome back, {username}. "
        if username
        else ""
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">
                BKAI STRUCTURAL VISION SYSTEM
            </div>
            <div class="hero-title">
                AI-Based Concrete Crack Inspection Platform
            </div>
            <div class="hero-subtitle">
                {welcome_text}Perform instance segmentation, crack geometry
                measurement, calibrated unit conversion, transparent assessment,
                and traceable PDF reporting in one integrated workspace.
            </div>
            <div class="hero-badges">
                <span class="hero-badge">Mask segmentation</span>
                <span class="hero-badge">Per-instance measurement</span>
                <span class="hero-badge">Calibration support</span>
                <span class="hero-badge">PDF reporting</span>
                <span class="hero-badge">Version {APP_VERSION}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_panel_start(
    title: str,
    subtitle: str = "",
) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def show_panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def show_status(
    text: str,
    status_type: str,
) -> None:
    st.markdown(
        f'<div class="status status-{status_type}">{text}</div>',
        unsafe_allow_html=True,
    )


def render_metric_cards(
    metric_items: List[Tuple[str, str, str]],
) -> None:
    for start_index in range(0, len(metric_items), 4):
        columns = st.columns(4)

        for offset, column in enumerate(columns):
            metric_index = start_index + offset

            if metric_index >= len(metric_items):
                continue

            label, value, note = metric_items[metric_index]

            with column:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# 12. PDF REPORT
# ============================================================

def pil_image_buffer(image: Image.Image) -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def build_pdf(
    inspection_id: str,
    image_name: str,
    original_image: Image.Image,
    segmented_image: Image.Image,
    measurement_image: Image.Image,
    results_table: pd.DataFrame,
    confidence_threshold: float,
    total_time: float,
    calibration_summary: str,
    assessment_summary: str,
) -> bytes:
    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475569"),
        )
    )

    story = []

    story.append(
        Paragraph(
            "BKAI CONCRETE CRACK INSPECTION REPORT",
            styles["Title"],
        )
    )
    story.append(Spacer(1, 4 * mm))

    metadata = [
        ["Inspection ID", inspection_id],
        ["Image file", image_name],
        ["Generated at", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Model", ROBOFLOW_MODEL],
        ["Confidence threshold", f"{confidence_threshold:.2f}"],
        ["End-to-end processing time", f"{total_time:.2f} s"],
        ["Calibration", calibration_summary],
    ]

    metadata_table = Table(
        metadata,
        colWidths=[55 * mm, 105 * mm],
    )
    metadata_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2ff")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(metadata_table)
    story.append(Spacer(1, 6 * mm))

    original_buffer = pil_image_buffer(original_image)
    segmented_buffer = pil_image_buffer(segmented_image)

    image_table = Table(
        [
            [
                Paragraph("<b>Original image</b>", styles["BodyText"]),
                Paragraph("<b>Segmentation result</b>", styles["BodyText"]),
            ],
            [
                RLImage(original_buffer, width=78 * mm, height=62 * mm),
                RLImage(segmented_buffer, width=78 * mm, height=62 * mm),
            ],
        ],
        colWidths=[82 * mm, 82 * mm],
    )

    image_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(image_table)
    story.append(PageBreak())

    story.append(
        Paragraph(
            "Quantitative measurements",
            styles["Heading1"],
        )
    )

    measurement_buffer = pil_image_buffer(measurement_image)
    story.append(
        RLImage(
            measurement_buffer,
            width=164 * mm,
            height=102 * mm,
        )
    )
    story.append(Spacer(1, 5 * mm))

    report_table_data = (
        [list(results_table.columns)]
        + results_table.astype(str).values.tolist()
    )

    report_table = Table(
        report_table_data,
        repeatRows=1,
        colWidths=[
            18 * mm,
            21 * mm,
            25 * mm,
            25 * mm,
            25 * mm,
            22 * mm,
            22 * mm,
        ],
    )

    report_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#f8fbff"),
                ]),
            ]
        )
    )

    story.append(report_table)
    story.append(Spacer(1, 6 * mm))

    story.append(
        Paragraph(
            "Assessment",
            styles["Heading1"],
        )
    )
    story.append(
        Paragraph(
            assessment_summary,
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            "Important limitation: the measurement result depends on image "
            "resolution, camera distance, perspective, segmentation accuracy, "
            "and calibration quality. This report does not replace an on-site "
            "assessment by a qualified structural engineer.",
            styles["ReportSmall"],
        )
    )

    document.build(story)
    output.seek(0)
    return output.getvalue()


# ============================================================
# 13. REFERENCE LIBRARY
# ============================================================

def show_engineering_reference() -> None:
    st.info(
        "This section is an engineering reference table. "
        "It is not an automatic crack-type classification result."
    )

    reference_data = [
        {
            "Component": "Beam",
            "Crack type": "Flexural crack",
            "Typical cause": "Bending demand, insufficient flexural capacity, or reinforcement deficiency.",
            "Typical appearance": "Often approximately vertical in the tension zone, commonly near mid-span.",
        },
        {
            "Component": "Beam",
            "Crack type": "Shear crack",
            "Typical cause": "High shear demand or inadequate shear reinforcement.",
            "Typical appearance": "Inclined crack, frequently developing near a support.",
        },
        {
            "Component": "Column",
            "Crack type": "Longitudinal splitting",
            "Typical cause": "High compression, inadequate confinement, or bond-related distress.",
            "Typical appearance": "One or more approximately vertical cracks.",
        },
        {
            "Component": "Slab",
            "Crack type": "Drying shrinkage crack",
            "Typical cause": "Moisture loss combined with restraint.",
            "Typical appearance": "Map pattern or relatively straight distributed cracks.",
        },
        {
            "Component": "Concrete wall",
            "Crack type": "Thermal crack",
            "Typical cause": "Temperature gradient and restrained deformation.",
            "Typical appearance": "Frequently vertical or distributed through the wall surface.",
        },
    ]

    st.dataframe(
        pd.DataFrame(reference_data),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 14. LOGIN AND REGISTRATION PAGE
# ============================================================

def show_authentication_page() -> None:
    show_hero()

    left_space, center_column, right_space = st.columns([1, 2.2, 1])

    with center_column:
        show_panel_start(
            "Secure workspace access",
            "Sign in to use the BKAI crack inspection workspace or create a new account.",
        )

        login_tab, register_tab = st.tabs(
            ["Sign in", "Create account"]
        )

        with login_tab:
            login_username = st.text_input(
                "Username",
                key="login_username",
            )
            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            if st.button(
                "Sign in",
                type="primary",
                use_container_width=True,
            ):
                if authenticate(
                    login_username,
                    login_password,
                ):
                    st.session_state.authenticated = True
                    st.session_state.username = login_username
                    st.session_state.profile_completed = True
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with register_tab:
            register_username = st.text_input(
                "Username",
                key="register_username",
            )
            register_email = st.text_input(
                "Email",
                key="register_email",
            )
            register_password = st.text_input(
                "Password",
                type="password",
                key="register_password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_password",
            )

            if st.button(
                "Create account",
                use_container_width=True,
            ):
                if (
                    not register_username
                    or not register_email
                    or not register_password
                ):
                    st.warning("Please complete all required fields.")
                elif "@" not in register_email or "." not in register_email:
                    st.error("Please provide a valid email address.")
                elif len(register_password) < 8:
                    st.error("Password must contain at least 8 characters.")
                elif register_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    successful, message = register_account(
                        register_username,
                        register_email,
                        register_password,
                    )

                    if successful:
                        st.success(message)
                    else:
                        st.error(message)

        show_panel_end()


# ============================================================
# 15. USER PROFILE
# ============================================================

def show_profile_form() -> bool:
    if st.session_state.get("profile_completed", False):
        return True

    show_panel_start(
        "User profile",
        "Complete the profile before starting an inspection.",
    )

    with st.form("profile_form"):
        first_column, second_column = st.columns(2)

        with first_column:
            full_name = st.text_input("Full name *")
            email = st.text_input("Email *")
            occupation = st.selectbox(
                "Occupation *",
                [
                    "Student",
                    "Graduate Student / Researcher",
                    "Structural Engineer",
                    "Site Engineer",
                    "Consultant",
                    "Contractor",
                    "Lecturer / Academic Staff",
                    "Other",
                ],
            )

        with second_column:
            organization = st.text_input("Organization / Company")
            country = st.text_input("Country / Region")
            project_name = st.text_input("Project / Case Name")

        purpose = st.selectbox(
            "Purpose of use",
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

        notes = st.text_area("Remarks / Notes")
        profile_submitted = st.form_submit_button(
            "Save profile",
        )

    if profile_submitted:
        if not full_name or not email:
            st.warning("Please complete all required fields.")
            show_panel_end()
            return False

        save_user_profile(
            {
                "username": st.session_state["username"],
                "full_name": full_name,
                "email": email,
                "occupation": occupation,
                "organization": organization,
                "country": country,
                "project_name": project_name,
                "purpose": purpose,
                "notes": notes,
            }
        )

        st.session_state.profile_completed = True
        st.success("Profile saved successfully.")
        st.rerun()

    show_panel_end()
    return False


# ============================================================
# 16. MAIN INSPECTION WORKSPACE
# ============================================================

def show_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=92)

        st.markdown(
            f"""
            <div class="sidebar-user">
                Active user<br>
                <span style="font-size:13px;font-weight:650;">
                    {st.session_state.get("username", "-")}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Detection settings")

        confidence_threshold = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
        )

        st.markdown("### Measurement calibration")

        use_scale = st.checkbox(
            "Convert pixels to millimetres",
            value=False,
        )

        millimetres_per_pixel = 1.0
        calibration_method = "Not configured"

        if use_scale:
            calibration_method = st.selectbox(
                "Calibration method",
                [
                    "Manual mm/pixel value",
                    "Reference ruler",
                    "Known object",
                    "Camera calibration",
                ],
            )

            millimetres_per_pixel = st.number_input(
                "Scale (mm/pixel)",
                min_value=0.0001,
                value=0.1000,
                step=0.0001,
                format="%.4f",
            )

        st.markdown("### Model information")
        st.caption(f"Model: {ROBOFLOW_MODEL}")
        st.caption(f"Application version: {APP_VERSION}")

        st.divider()

        if st.button(
            "Log out",
            use_container_width=True,
        ):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.profile_completed = False
            st.rerun()

    return {
        "confidence_threshold": confidence_threshold,
        "use_scale": use_scale,
        "millimetres_per_pixel": millimetres_per_pixel,
        "calibration_method": calibration_method,
    }


def process_uploaded_image(
    uploaded_file,
    file_index: int,
    settings: Dict[str, Any],
) -> None:
    inspection_id = (
        f"BKAI-{dt.datetime.now():%Y%m%d-%H%M%S}-{file_index:03d}"
    )

    st.markdown(
        f'<div class="section-heading">{inspection_id}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <span class="inspection-chip">File: {uploaded_file.name}</span>
        <span class="inspection-chip">Model: {ROBOFLOW_MODEL}</span>
        <span class="inspection-chip">
            Threshold: {settings["confidence_threshold"]:.2f}
        </span>
        """,
        unsafe_allow_html=True,
    )

    try:
        original_image = validate_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    image_width, image_height = original_image.size

    encoded_image = io.BytesIO()
    original_image.save(
        encoded_image,
        format="JPEG",
        quality=95,
    )

    start_time = time.perf_counter()

    try:
        with st.spinner("Running AI inference..."):
            inference_result = run_inference(
                encoded_image.getvalue(),
                ROBOFLOW_API_KEY,
                ROBOFLOW_MODEL,
            )
    except requests.RequestException as exc:
        st.error(f"Inference service error: {exc}")
        return
    except ValueError as exc:
        st.error(f"Invalid API response: {exc}")
        return

    total_time = time.perf_counter() - start_time

    raw_predictions = inference_result.get("predictions", [])

    predictions = [
        prediction
        for prediction in raw_predictions
        if to_float(prediction.get("confidence"))
        >= settings["confidence_threshold"]
        and extract_polygons(
            prediction.get("points"),
            image_width,
            image_height,
        )
    ]

    if not predictions:
        original_column, result_column = st.columns(2)

        with original_column:
            st.subheader("Original image")
            st.image(
                original_image,
                use_container_width=True,
            )

        with result_column:
            st.subheader("Analysis result")
            st.image(
                original_image,
                use_container_width=True,
            )
            show_status(
                "No crack instance passed the selected confidence threshold.",
                "success",
            )

        return

    segmented_image = draw_segmentation_overlay(
        original_image,
        predictions,
    )
    measurement_image = draw_measurement_overlay(
        segmented_image,
        predictions,
    )

    original_column, segmented_column = st.columns(2)

    with original_column:
        st.subheader("Original image")
        st.image(
            original_image,
            use_container_width=True,
        )

    with segmented_column:
        st.subheader("Segmentation result")
        st.image(
            segmented_image,
            use_container_width=True,
        )
        show_status(
            f"{len(predictions)} crack instance(s) were detected.",
            "warning",
        )

    crack_results = [
        create_crack_result(
            prediction=prediction,
            crack_index=index,
            image_width=image_width,
            image_height=image_height,
            use_scale=settings["use_scale"],
            millimetres_per_pixel=settings[
                "millimetres_per_pixel"
            ],
        )
        for index, prediction in enumerate(
            predictions,
            start=1,
        )
    ]

    raw_results = pd.DataFrame(
        [asdict(result) for result in crack_results]
    )

    length_unit = crack_results[0].length_unit

    result_table = pd.DataFrame(
        {
            "Crack ID": raw_results["crack_id"],
            "Confidence": raw_results["confidence"].map(
                lambda value: f"{value:.3f}"
            ),
            f"Length ({length_unit})":
                raw_results["length_value"].map(
                    lambda value: f"{value:.2f}"
                ),
            f"Avg width ({length_unit})":
                raw_results["avg_width_value"].map(
                    lambda value: f"{value:.2f}"
                ),
            f"Max width ({length_unit})":
                raw_results["max_width_value"].map(
                    lambda value: f"{value:.2f}"
                ),
            "Orientation (°)":
                raw_results["orientation_deg"].map(
                    lambda value: f"{value:.1f}"
                ),
            "Tortuosity":
                raw_results["tortuosity"].map(
                    lambda value: f"{value:.3f}"
                ),
        }
    )

    combined_mask = union_prediction_mask(
        predictions,
        image_width,
        image_height,
    )

    total_area_px2 = float(
        cv2.countNonZero(combined_mask)
    )
    total_area_ratio = (
        total_area_px2
        / float(image_width * image_height)
        * 100.0
    )

    average_confidence = float(
        raw_results["confidence"].mean()
    )
    total_length = float(
        raw_results["length_value"].sum()
    )
    average_width = float(
        raw_results["avg_width_value"].mean()
    )
    maximum_width = float(
        raw_results["max_width_value"].max()
    )

    metric_items = [
        (
            "Cracks detected",
            str(len(predictions)),
            "Number of segmentation instances above the threshold.",
        ),
        (
            "Average confidence",
            f"{average_confidence * 100:.1f}%",
            "Mean model confidence of the retained instances.",
        ),
        (
            f"Total length ({length_unit})",
            f"{total_length:.2f}",
            "Sum of weighted skeleton lengths.",
        ),
        (
            f"Average width ({length_unit})",
            f"{average_width:.2f}",
            "Mean distance-transform width across instances.",
        ),
        (
            f"Maximum width ({length_unit})",
            f"{maximum_width:.2f}",
            "Largest estimated local crack width.",
        ),
        (
            "Crack area ratio",
            f"{total_area_ratio:.2f}%",
            "Union crack-mask area divided by image area.",
        ),
        (
            "Image crack extent",
            classify_image_extent(total_area_ratio),
            "Image-space extent only; not structural severity.",
        ),
        (
            "End-to-end time",
            f"{total_time:.2f} s",
            "Network request, inference response, and client processing.",
        ),
    ]

    st.markdown(
        '<div class="section-heading">Inspection summary</div>',
        unsafe_allow_html=True,
    )
    render_metric_cards(metric_items)

    analysis_tab, table_tab, assessment_tab, report_tab = st.tabs(
        [
            "Measurement view",
            "Instance table",
            "Assessment",
            "Report",
        ]
    )

    with analysis_tab:
        st.image(
            measurement_image,
            use_container_width=True,
        )
        st.caption(
            "Yellow: estimated skeleton centerline. "
            "Green: skeleton endpoints. "
            "Red: maximum distance-transform location."
        )

    with table_tab:
        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Export measurements as CSV",
            data=result_table.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=f"{inspection_id}_measurements.csv",
            mime="text/csv",
            key=f"csv_{inspection_id}",
        )

    if settings["use_scale"]:
        width_grade = research_width_grade(
            maximum_width
        )

        assessment_summary = (
            f"The maximum calibrated crack width is "
            f"{maximum_width:.3f} mm. Using the currently "
            f"configured research demonstration thresholds, "
            f"the result is classified as {width_grade}. "
            f"The thresholds must be replaced or justified "
            f"using the specific engineering standard adopted "
            f"in the manuscript."
        )

        calibration_summary = (
            f'{settings["calibration_method"]}; '
            f'{settings["millimetres_per_pixel"]:.4f} mm/pixel'
        )
    else:
        assessment_summary = (
            f"The detected crack mask occupies "
            f"{total_area_ratio:.3f}% of the image, corresponding "
            f"to {classify_image_extent(total_area_ratio).lower()} "
            f"image-space extent. Physical calibration is not "
            f"configured; therefore, structural severity cannot "
            f"be inferred from pixel measurements alone."
        )

        calibration_summary = (
            "Not configured; pixel-space measurement only"
        )

    with assessment_tab:
        if settings["use_scale"]:
            show_status(
                assessment_summary,
                "warning",
            )
        else:
            show_status(
                assessment_summary,
                "danger",
            )

        with st.expander(
            "Engineering reference library",
            expanded=False,
        ):
            show_engineering_reference()

    with report_tab:
        pdf_bytes = build_pdf(
            inspection_id=inspection_id,
            image_name=uploaded_file.name,
            original_image=original_image,
            segmented_image=segmented_image,
            measurement_image=measurement_image,
            results_table=result_table,
            confidence_threshold=settings[
                "confidence_threshold"
            ],
            total_time=total_time,
            calibration_summary=calibration_summary,
            assessment_summary=assessment_summary,
        )

        st.download_button(
            "Download PDF inspection report",
            data=pdf_bytes,
            file_name=f"{inspection_id}.pdf",
            mime="application/pdf",
            key=f"pdf_{inspection_id}",
            use_container_width=True,
        )

        st.caption(
            "The PDF includes inspection metadata, original and "
            "segmented images, per-instance measurements, assessment, "
            "calibration status, and methodological limitations."
        )


def show_main_workspace() -> None:
    if not ROBOFLOW_API_KEY:
        st.error(
            "ROBOFLOW_API_KEY is not configured. "
            "Open Streamlit Cloud → App settings → Secrets, "
            "then add the Roboflow key."
        )
        st.stop()

    settings = show_sidebar()
    show_hero(st.session_state.get("username", ""))

    if not show_profile_form():
        return

    show_panel_start(
        "New concrete crack inspection",
        "Upload one or more JPG or PNG images. "
        "Each detected crack instance will be measured separately.",
    )

    uploaded_files = st.file_uploader(
        "Upload concrete images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    analyze_button = st.button(
        "Analyze uploaded images",
        type="primary",
        use_container_width=True,
    )

    show_panel_end()

    if not analyze_button:
        return

    if not uploaded_files:
        st.warning("Please upload at least one image.")
        return

    for file_index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        st.divider()

        process_uploaded_image(
            uploaded_file,
            file_index,
            settings,
        )


# ============================================================
# 17. APPLICATION START
# ============================================================

def main() -> None:
    initialize_database()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if "profile_completed" not in st.session_state:
        st.session_state.profile_completed = False

    if st.session_state.authenticated:
        show_main_workspace()
    else:
        show_authentication_page()


if __name__ == "__main__":
    main()
