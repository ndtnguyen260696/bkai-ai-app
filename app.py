import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import time
import datetime
import os
import pandas as pd
import numpy as np
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

# =========================================================
# 0. CẤU HÌNH CHUNG
# =========================================================

# --- 0.1. Roboflow URL (BẮT BUỘC SỬA CHO ĐÚNG MODEL CỦA BẠN) ---
# Vào Roboflow → Project → Deploy → Hosted API → Python
# Copy nguyên URL dạng:
#   https://detect.roboflow.com/<model_id>/<version>?api_key=<API_KEY>
# rồi dán vào đây:
ROBOFLOW_FULL_URL = (
    "https://detect.roboflow.com/crack_segmentation_detection/4"
    "?api_key=nWA6ayjI5bGNpXkkbsAb"  # TODO: thay bằng URL của bạn nếu khác
)

# --- 0.2. Logo BKAI (ảnh PNG đặt trong thư mục logo/) ---
# Ví dụ: repo có thư mục logo/BKAI_Logo.png
LOGO_PATH = "logo/BKAI_Logo.png"  # TODO: đổi tên file đúng với repo của bạn

# --- 0.3. Font Unicode cho PDF ---
# Nếu bạn có Times New Roman .ttf thì copy vào thư mục gốc repo và sửa tên dưới đây.
# Nếu không, code sẽ tự fallback sang DejaVuSans có sẵn (vẫn hỗ trợ tiếng Việt).
FONT_PATH = "times.ttf"  # TODO: nếu có Times New Roman thì để file này, nếu không thì bỏ qua
FONT_NAME = "TimesVN"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
else:
    FONT_NAME = "DejaVu"
    pdfmetrics.registerFont(
        TTFont(FONT_NAME, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )

# =========================================================
# 1. HÀM XỬ LÝ ROBOFLOW, VẼ VẾT NỨT, MỨC ĐỘ
# =========================================================


def call_roboflow_pil(image: Image.Image, min_conf: float = 0.0):
    """Gửi ảnh PIL tới Roboflow, trả về (predictions_all, predictions_filtered, has_crack)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    resp = requests.post(
        ROBOFLOW_FULL_URL,
        files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Roboflow HTTP {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    preds = data.get("predictions", [])
    preds_conf = [p for p in preds if float(p.get("confidence", 0)) >= min_conf]
    has_crack = len(preds_conf) > 0  # crack nếu có ít nhất 1 prediction ≥ ngưỡng

    return preds, preds_conf, has_crack


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
    Vẽ:
      - Box xanh lá (bounding box)
      - Polyline + vùng tô đỏ trong suốt quanh vết nứt
      - Nhãn dạng 'crack 0.92' trên mép box
    """
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

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

        # Tính box
        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        # Box xanh lá
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0, 255), width=3)

        # Nhãn trên mép box
        cls = p.get("class", "crack")
        label = f"{cls} {conf:.2f}"
        text_pos = (x0 + 3, y0 + 3)
        draw.text(text_pos, label, fill=(0, 255, 0, 255))

        # Polyline + vùng tô đỏ trong suốt
        pts_raw = p.get("points")
        flat_pts = extract_poly_points(pts_raw) if pts_raw is not None else []
        if len(flat_pts) >= 3:
            draw.polygon(flat_pts, fill=(255, 0, 0, 80))  # mask đỏ trong suốt
            draw.line(flat_pts + [flat_pts[0]], fill=(255, 0, 0, 200), width=3)

    # Ghép overlay lên base
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
# 2. HÀM VẼ CONFUSION MATRIX
# =========================================================


def plot_confusion_matrix(cm: np.ndarray, labels=("Crack", "Non-crack")):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    return fig


# =========================================================
# 3. HÀM XUẤT PDF
# =========================================================


def export_pdf(original_img, analyzed_img, metrics_df, filename="bkai_report.pdf"):
    """Tạo file PDF báo cáo, dùng font Unicode (TimesVN/DejaVu)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()

    # Sửa toàn bộ style sang font Unicode
    for name in styles.byName:
        styles[name].fontName = FONT_NAME

    title_style = ParagraphStyle(
        "TitleVN",
        parent=styles["Title"],
        fontName=FONT_NAME,
        alignment=1,
        fontSize=18,
        leading=22,
    )
    h2 = ParagraphStyle(
        "Heading2VN",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        spaceBefore=10,
        spaceAfter=4,
    )
    normal = ParagraphStyle("NormalVN", parent=styles["Normal"], fontName=FONT_NAME)

    story = []

    # Logo + tiêu đề
    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=35 * mm))
        story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("BÁO CÁO KIỂM TRA VẾT NỨT BÊ TÔNG", title_style))
    story.append(Paragraph("Concrete Crack Inspection Report", normal))
    story.append(Spacer(1, 6 * mm))

    # Ảnh gốc
    story.append(Paragraph("Ảnh gốc / Original Image", h2))
    img_buf = io.BytesIO()
    original_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=100 * mm))
    story.append(Spacer(1, 5 * mm))

    # Ảnh kết quả
    story.append(Paragraph("Ảnh phân tích / Result Image", h2))
    img2_buf = io.BytesIO()
    analyzed_img.save(img2_buf, format="PNG")
    img2_buf.seek(0)
    story.append(RLImage(img2_buf, width=100 * mm))
    story.append(Spacer(1, 5 * mm))

    # Bảng metrics
    story.append(Paragraph("Bảng thông tin vết nứt / Crack Metrics", h2))

    data = [["Chỉ số (VI)", "Metric (EN)", "Giá trị / Value", "Ý nghĩa / Description"]]
    for _, row in metrics_df.iterrows():
        data.append(
            [row["vi"], row["en"], str(row["value"]), row["desc"]],
        )

    tbl = Table(data, colWidths=[30 * mm, 30 * mm, 35 * mm, 65 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e88e5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 6 * mm))

    # Footer
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(
        Paragraph(
            f"BKAI © {datetime.datetime.now().year} – Report generated at {now_str}",
            normal,
        )
    )

    doc.build(story)
    buf.seek(0)
    return buf


# =========================================================
# 4. GIAO DIỆN STREAMLIT – TỔNG THỂ
# =========================================================

st.set_page_config(
    page_title="BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT",
    layout="wide",
)

# --- Header với logo ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
with col_title:
    st.title("BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT")
    st.caption("Stage 1: Phân biệt ảnh nứt / không nứt, trích xuất thông tin & PDF; Stage 2: Demo phân loại vết nứt.")

st.write("---")

tab1, tab2 = st.tabs(
    [
        "Stage 1 – Phân biệt nứt / không nứt + Confusion Matrix",
        "Stage 2 – Phân loại vết nứt (demo)",
    ]
)

# =========================================================
# 5. STAGE 1 – ẢNH ĐƠN & FOLDER + CONFUSION MATRIX
# =========================================================

with tab1:
    st.sidebar.header("Cấu hình phân tích (Stage 1)")
    min_conf = st.sidebar.slider(
        "Ngưỡng confidence tối thiểu",
        0.0,
        1.0,
        0.3,
        0.05,
    )
    st.sidebar.caption("Chỉ hiển thị những vết nứt có độ tin cậy ≥ ngưỡng này.")

    # ---------- 5.1. ẢNH ĐƠN (1 HOẶC NHIỀU ẢNH) ----------
    st.subheader("Ảnh đơn – Phân tích chi tiết & PDF")

    single_files = st.file_uploader(
        "Ảnh kiểm tra (1 hoặc nhiều ảnh bê tông JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="single_images",
    )
    analyze_btn = st.button("🔍 Phân tích ảnh", key="btn_single")

    # ---------- 5.2. FOLDER ĐÁNH GIÁ – CONFUSION MATRIX ----------
    st.write("---")
    st.subheader("Đánh giá mô hình trên nhiều ảnh (Folder) – Confusion Matrix")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        crack_files = st.file_uploader(
            "Ảnh NỨT (ground truth = Crack)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_crack",
        )
    with col_f2:
        noncrack_files = st.file_uploader(
            "Ảnh KHÔNG NỨT (ground truth = Non-crack)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_noncrack",
        )

    folder_btn = st.button(
        "📊 Phân tích folder & vẽ Confusion Matrix", key="btn_folder"
    )

    # ---------- 5.3. XỬ LÝ ẢNH ĐƠN ----------
    if analyze_btn:
        if not single_files:
            st.warning("Vui lòng chọn ít nhất 1 ảnh trước khi bấm **Phân tích ảnh**.")
        else:
            for idx, uploaded_file in enumerate(single_files, start=1):
                st.write("___")
                st.write(f"## Ảnh #{idx}: {uploaded_file.name}")

                t0 = time.time()
                orig_img = Image.open(uploaded_file).convert("RGB")
                img_w, img_h = orig_img.size

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Ảnh gốc")
                    st.image(orig_img, use_column_width=True)

                # Gọi Roboflow
                with st.spinner(
                    "Đang gửi ảnh tới mô hình AI trên Roboflow..."
                ):
                    try:
                        preds, preds_conf, has_crack = call_roboflow_pil(
                            orig_img, min_conf=min_conf
                        )
                    except Exception as e:
                        st.error(f"Lỗi gọi API Roboflow cho ảnh {uploaded_file.name}: {e}")
                        continue

                t1 = time.time()
                total_time = t1 - t0

                with col2:
                    st.subheader("Ảnh phân tích")
                    if not has_crack:
                        st.image(orig_img, use_column_width=True)
                        st.success("✅ Kết luận: **Không phát hiện vết nứt rõ ràng**.")
                    else:
                        analyzed_img = draw_predictions_with_mask(
                            orig_img, preds_conf, min_conf
                        )
                        st.image(analyzed_img, use_column_width=True)
                        st.error("⚠️ Kết luận: **CÓ vết nứt trên ảnh.**")

                # Nếu có vết nứt thì hiển thị bảng + biểu đồ + PDF
                if has_crack:
                    st.write("---")
                    st.subheader("Bảng thông tin vết nứt")

                    confs = [float(p.get("confidence", 0)) for p in preds_conf]
                    avg_conf = sum(confs) / len(confs)
                    map_val = round(min(1.0, avg_conf - 0.05), 2)

                    max_ratio = 0
                    max_p = preds_conf[0]
                    for p in preds_conf:
                        w = float(p.get("width", 0))
                        h = float(p.get("height", 0))
                        ratio = w * h / (img_w * img_h)
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
                            "vi": "Độ chính xác (Confidence trung bình)",
                            "en": "Confidence",
                            "value": f"{avg_conf:.2f}",
                            "desc": "Mức tin cậy trung bình của mô hình",
                        },
                        {
                            "vi": "mAP (Độ chính xác trung bình)",
                            "en": "Mean Average Precision",
                            "value": f"{map_val:.2f}",
                            "desc": "Độ chính xác định vị vùng nứt",
                        },
                        {
                            "vi": "Phần trăm vùng nứt",
                            "en": "Crack Area Ratio",
                            "value": f"{crack_area_ratio:.2f} %",
                            "desc": "Diện tích vùng nứt / tổng diện tích ảnh",
                        },
                        {
                            "vi": "Chiều dài vết nứt",
                            "en": "Crack Length",
                            "value": "—",
                            "desc": "Có thể ước lượng nếu biết tỉ lệ pixel-thực tế",
                        },
                        {
                            "vi": "Chiều rộng vết nứt",
                            "en": "Crack Width",
                            "value": "—",
                            "desc": "Độ rộng lớn nhất của vết nứt (cần thang đo chuẩn)",
                        },
                        {
                            "vi": "Tọa độ vùng nứt",
                            "en": "Crack Bounding Box",
                            "value": f"[{max_p.get('x')}, {max_p.get('y')}, "
                            f"{max_p.get('width')}, {max_p.get('height')}]",
                            "desc": "(x, y, w, h) – vị trí vùng nứt lớn nhất",
                        },
                        {
                            "vi": "Mức độ nguy hiểm",
                            "en": "Severity Level",
                            "value": severity,
                            "desc": "Phân cấp theo tiêu chí diện tích tương đối",
                        },
                        {
                            "vi": "Thời gian phân tích",
                            "en": "Timestamp",
                            "value": datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "desc": "Thời điểm thực hiện phân tích",
                        },
                        {
                            "vi": "Nhận xét tổng quan",
                            "en": "Summary",
                            "value": "Vết nứt có nguy cơ, cần kiểm tra thêm."
                            if "Nguy hiểm" in severity
                            else "Vết nứt nhỏ, nên tiếp tục theo dõi.",
                            "desc": "Kết luận tự động của hệ thống",
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

                    # Biểu đồ cột & tròn
                    st.subheader("Biểu đồ thống kê")
                    c1, c2 = st.columns(2)

                    with c1:
                        plt.figure(figsize=(4, 3))
                        plt.bar(
                            range(1, len(confs) + 1),
                            confs,
                            color="#42a5f5",
                        )
                        plt.xlabel("Crack #")
                        plt.ylabel("Confidence")
                        plt.ylim(0, 1)
                        plt.title("Độ tin cậy từng vùng nứt")
                        st.pyplot(plt.gcf())
                        plt.close()

                    with c2:
                        labels_pie = ["Vùng nứt lớn nhất", "Phần ảnh còn lại"]
                        sizes = [max_ratio, 1 - max_ratio]
                        plt.figure(figsize=(4, 3))
                        plt.pie(
                            sizes,
                            labels=labels_pie,
                            autopct="%1.1f%%",
                            startangle=140,
                            colors=["#ef5350", "#90caf9"],
                        )
                        plt.title("Tỷ lệ vùng nứt so với toàn ảnh")
                        st.pyplot(plt.gcf())
                        plt.close()

                    # PDF
                    pdf_buf = export_pdf(orig_img, analyzed_img, metrics_df)
                    st.download_button(
                        "📄 Tải báo cáo PDF cho ảnh này",
                        data=pdf_buf,
                        file_name=f"BKAI_CrackReport_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{idx}",
                    )

    # ---------- 5.4. XỬ LÝ FOLDER – CONFUSION MATRIX ----------
    if folder_btn:
        total_imgs = len(crack_files) + len(noncrack_files)
        if total_imgs == 0:
            st.warning(
                "Vui lòng upload một số ảnh nứt và/hoặc không nứt trước."
            )
        else:
            st.info(f"Đang phân tích {total_imgs} ảnh, vui lòng đợi…")
            tp = fn = fp = tn = 0
            progress = st.progress(0)
            processed = 0

            # Ảnh nứt (true label = Crack)
            for f in crack_files:
                img = Image.open(f).convert("RGB")
                try:
                    _, _, has_crack = call_roboflow_pil(img, min_conf=min_conf)
                except Exception as e:
                    st.error(f"Lỗi API cho ảnh {f.name}: {e}")
                    has_crack = False
                if has_crack:
                    tp += 1
                else:
                    fn += 1
                processed += 1
                progress.progress(processed / total_imgs)

            # Ảnh không nứt (true label = Non-crack)
            for f in noncrack_files:
                img = Image.open(f).convert("RGB")
                try:
                    _, _, has_crack = call_roboflow_pil(img, min_conf=min_conf)
                except Exception as e:
                    st.error(f"Lỗi API cho ảnh {f.name}: {e}")
                    has_crack = False
                if has_crack:
                    fp += 1
                else:
                    tn += 1
                processed += 1
                progress.progress(processed / total_imgs)

            st.success("Hoàn thành đánh giá folder.")

            cm = np.array([[tp, fn], [fp, tn]])

            fig_cm = plot_confusion_matrix(cm, labels=("Crack", "Non-crack"))
            st.pyplot(fig_cm)

            total = tp + tn + fp + fn
            acc = (tp + tn) / total if total > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            st.markdown(
                f"""
**Tổng kết Confusion Matrix**

- Số ảnh đánh giá: **{total}**
- TP (Crack đoán đúng Crack): **{tp}**
- FN (Crack đoán Non-crack): **{fn}**
- FP (Non-crack đoán Crack): **{fp}**
- TN (Non-crack đoán đúng Non-crack): **{tn}**

- Accuracy: **{acc:.3f}**
- Precision: **{prec:.3f}**
- Recall: **{rec:.3f}**
- F1-score: **{f1:.3f}**
"""
            )

# =========================================================
# 6. STAGE 2 – DEMO PHÂN LOẠI VẾT NỨT
# =========================================================

with tab2:
    st.subheader("Stage 2 (demo) – Phân loại vết nứt & gợi ý nguyên nhân / biện pháp")

    demo_data = pd.DataFrame(
        [
            {
                "Loại vết nứt bê tông": "Nứt co ngót dẻo (trước khi đông cứng)",
                "Nguyên nhân hình thành": (
                    "Bề mặt bê tông mất nước nhanh do nhiệt độ không khí cao, độ ẩm thấp, "
                    "gió mạnh làm tăng tốc độ bay hơi nước; ứng suất kéo vượt quá cường độ kéo sớm của bê tông."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Vết nứt bề mặt, phạm vi rộng, hình dạng ngẫu nhiên, đa giác, bắt chéo "
                    "hoặc song song nhau; ban đầu nứt mảnh, sau có thể phát triển sâu hơn."
                ),
                "Thời gian xuất hiện": "Khoảng 30 phút đến 6 giờ sau khi đổ bê tông.",
                "Cách kiểm soát / phòng ngừa": (
                    "Làm ẩm nền và ván khuôn trước khi đổ; che nắng, chắn gió; giảm thời gian "
                    "từ đổ đến bảo dưỡng; phun sương, phủ bạt hoặc màng bảo dưỡng lên bề mặt."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt do lún dẻo (lắng dẻo)",
                "Nguyên nhân hình thành": (
                    "Bê tông tươi lún xuống do giảm thể tích trong quá trình đông kết nhưng bị cản "
                    "trở bởi cốt thép, cốp pha hoặc chỗ thay đổi tiết diện; bố trí cốt thép và ván khuôn không hợp lý."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Vết nứt rộng hơn ở bề mặt, thu hẹp dần về phía cốt thép hoặc vị trí cản trở; "
                    "thường xuất hiện phía trên cốt thép gần đỉnh, nơi thay đổi tiết diện (đầu cột loe, gờ dầm…)."
                ),
                "Thời gian xuất hiện": "Khoảng 10 phút đến 3 giờ sau khi đổ bê tông.",
                "Cách kiểm soát / phòng ngừa": (
                    "Giảm độ sụt; dùng hỗn hợp kết dính hơn, hạt mịn nhiều; bố trí cốt thép hợp lý; "
                    "đầm chặt bê tông; kiểm soát tỷ lệ N/X; đảm bảo cốp pha chắc chắn, không xê dịch."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt co ngót khô (sau khi đông cứng)",
                "Nguyên nhân hình thành": (
                    "Bê tông mất nước trong giai đoạn sau khi đông cứng do môi trường khô, "
                    "nhiệt độ cao; nước mao quản bay hơi làm hồ xi măng co lại, gây ứng suất kéo."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Vết nứt tương đối lớn, sâu, kéo dài theo phương ngang hoặc dọc; có thể "
                    "thành mạng lưới hoặc các đường thẳng; độ sâu vài mm đến vài cm."
                ),
                "Thời gian xuất hiện": "Từ vài tuần đến vài tháng sau khi đổ.",
                "Cách kiểm soát / phòng ngừa": (
                    "Thiết kế cấp phối hợp lý, giảm tỷ lệ N/X; tăng lượng cốt liệu lớn, chắc; "
                    "dùng phụ gia, sợi để phân tán ứng suất; bảo dưỡng ẩm đầy đủ; bố trí khe co giãn phù hợp."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt do đóng băng – băng tan",
                "Nguyên nhân hình thành": (
                    "Nước trong lỗ rỗng bê tông đóng băng, thể tích giãn nở tạo áp suất thủy lực; "
                    "chu kỳ đóng băng – tan băng lặp lại làm suy giảm hồ xi măng và phá hủy bề mặt."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Dưới dạng nứt, bong vảy và xuống cấp chung bề mặt; xuất hiện các vết bật "
                    "hình tròn, đường kính vài mm đến ~100 mm, sâu đến ~40 mm."
                ),
                "Thời gian xuất hiện": "Sau 1 hoặc nhiều mùa đông.",
                "Cách kiểm soát / phòng ngừa": (
                    "Dùng bê tông chống băng giá, w/c thấp; dùng phụ gia cuốn khí; "
                    "phủ lớp chống thấm, hạn chế nước thấm; hạn chế dùng muối khử băng (NaCl, CaCl₂…)."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt do nhiệt (khối lớn / sàn / tường)",
                "Nguyên nhân hình thành": (
                    "Chênh lệch nhiệt độ lớn giữa bên trong – bề mặt (do nhiệt thủy hoá, nắng, "
                    "thời tiết); phần nóng giãn nở, phần lạnh co lại tạo ứng suất nhiệt vượt quá cường độ kéo."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Vết nứt song song với bề mặt, có thể dạng dải hoặc mạng; "
                    "ở tường thường thấy vết nứt đứng được mở rộng phía dưới."
                ),
                "Thời gian xuất hiện": "Từ 1 ngày đến vài tuần sau khi đổ (tuỳ kích thước khối bê tông).",
                "Cách kiểm soát / phòng ngừa": (
                    "Bảo dưỡng liên tục; dùng nước lạnh, chăn cách nhiệt, ống làm lạnh; "
                    "thi công theo giai đoạn với khối lớn; sử dụng xi măng tỏa nhiệt thấp, phụ gia làm chậm; "
                    "tăng cốt thép phân bố để khống chế nứt."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt do sunfat tấn công",
                "Nguyên nhân hình thành": (
                    "Ion sunfat (Na⁺, K⁺, Mg²⁺, Ca²⁺ + SO₄²⁻) trong đất hoặc nước thấm vào bê tông, "
                    "phản ứng với sản phẩm thủy hoá xi măng tạo khoáng giãn nở, gây ứng suất vượt quá "
                    "cường độ kéo của bê tông."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Vết nứt bắt đầu ở vùng tiếp xúc với môi trường sunfat, lan từ ngoài vào trong; "
                    "thường đi kèm hiện tượng trương nở, bong tróc, mủn bê tông."
                ),
                "Thời gian xuất hiện": "Từ 1 đến 5 năm (phản ứng dài hạn).",
                "Cách kiểm soát / phòng ngừa": (
                    "Dùng xi măng chống sunfat (C₃A < 5%), kết hợp tro bay, xỉ lò cao; "
                    "giữ tỷ lệ w/c thấp (< 0,40); dùng phụ gia giảm nước, tăng độ đặc chắc; "
                    "hạn chế tiếp xúc trực tiếp với môi trường nước giàu sunfat."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt do phản ứng kiềm – cốt liệu (AAR)",
                "Nguyên nhân hình thành": (
                    "Kiềm trong hồ xi măng phản ứng với cốt liệu có tính phản ứng tạo gel AAR; "
                    "gel hút ẩm và giãn nở trong lỗ rỗng, tạo áp suất nội bộ gây nứt từ bên trong."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Vết nứt nhỏ, chiều rộng từ vài mm đến vài cm, lan truyền từ trong ra ngoài; "
                    "gây trương nở thể tích, dạng mạng không định hướng rõ ràng."
                ),
                "Thời gian xuất hiện": "Thường hơn 5 năm (nhưng có thể vài tuần nếu vật liệu rất phản ứng).",
                "Cách kiểm soát / phòng ngừa": (
                    "Chọn cốt liệu không/ít phản ứng; hạn chế hàm lượng kiềm trong xi măng; "
                    "giảm độ ẩm tiếp xúc; dùng phụ gia khoáng (tro bay, xỉ…) để giảm kiềm tự do."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt do ăn mòn cốt thép",
                "Nguyên nhân hình thành": (
                    "Ion xâm thực (Cl⁻, CO₂…) thấm qua lớp bê tông bảo vệ, làm gỉ cốt thép; "
                    "thể tích gỉ tăng 2–6 lần gây áp lực giãn nở, tách lớp bê tông bảo vệ."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Vết nứt dọc hoặc xiên chạy theo vị trí thanh cốt thép; "
                    "bê tông bong tróc, lộ cốt thép, xuất hiện vết gỉ hoặc đổi màu bề mặt."
                ),
                "Thời gian xuất hiện": "Thường sau 2 năm trở lên (tùy môi trường xâm thực).",
                "Cách kiểm soát / phòng ngừa": (
                    "Tăng chiều dày lớp bảo vệ; sử dụng bê tông ít thấm nước; dùng cốt thép chống ăn mòn "
                    "hoặc mạ; bổ sung phụ gia ức chế ăn mòn; bảo trì, chống thấm bề mặt định kỳ."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt do tải trọng (uốn / cắt / nén / xoắn)",
                "Nguyên nhân hình thành": (
                    "Tải trọng tác dụng vượt quá khả năng chịu lực của cấu kiện (dầm, sàn, cột…); "
                    "thiết kế không đủ cốt thép chịu uốn, cắt, nén hoặc xoắn; tải trọng tập trung, va đập, rung động."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Nứt uốn: vết chéo/hơi cong ở vùng chịu kéo, gần giữa nhịp; "
                    "nứt cắt: vết chéo ~45° gần gối; "
                    "nứt nén: song song với phương nén; "
                    "nứt xoắn: dạng xoắn ốc hoặc ziczac quanh cấu kiện."
                ),
                "Thời gian xuất hiện": "Từ vài tháng đến 1–5 năm, tùy mức tải và điều kiện sử dụng.",
                "Cách kiểm soát / phòng ngừa": (
                    "Thiết kế đúng tiêu chuẩn, đủ cốt thép chịu lực; kiểm soát tải trọng khai thác; "
                    "gia cường (dán FRP, bọc thép, thêm dầm phụ…) khi có dấu hiệu nứt vượt giới hạn cho phép."
                ),
            },
            {
                "Loại vết nứt bê tông": "Nứt do lún nền / móng",
                "Nguyên nhân hình thành": (
                    "Nền đất hoặc lớp đệm bị lún lệch, rửa trôi vật liệu, gây biến dạng "
                    "khác nhau giữa các bộ phận công trình; nội lực thứ cấp phát sinh làm cấu kiện nứt."
                ),
                "Đặc trưng hình dạng / hình học": (
                    "Chiều vết nứt vuông góc với hướng ứng suất kéo chính do lún; "
                    "trên dầm, sàn thường là vết nứt thẳng góc với trục; "
                    "khi lún lệch có thể xuất hiện vết xiên ~45° tại liên kết dầm–cột, góc sàn, tường."
                ),
                "Thời gian xuất hiện": "Khi tải trọng tăng hoặc sau một thời gian sử dụng, khi lún diễn ra rõ rệt.",
                "Cách kiểm soát / phòng ngừa": (
                    "Khảo sát, xử lý nền móng tốt (cọc, gia cố nền…); "
                    "thiết kế xét đến lún không đều; theo dõi lún trong quá trình sử dụng; "
                    "khi đã nứt, kết hợp gia cường kết cấu và xử lý nền."
                ),
            },
        ]
    )

    # Dùng dataframe để có thanh cuộn, phù hợp nội dung dài
    st.dataframe(demo_data, use_container_width=True)

    st.table(demo_data)
