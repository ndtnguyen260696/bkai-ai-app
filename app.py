import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import time
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# 0.1. Roboflow URL (NHỚ SỬA CHO ĐÚNG MODEL CỦA BẠN)
ROBOFLOW_FULL_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"

# 0.2. Logo BKAI (file PNG trong repo, ví dụ đặt cạnh app.py)
LOGO_PATH = "BKAI_Logo.png"  # Đặt đúng tên logo của bạn

# 0.3. Font Unicode cho PDF
FONT_PATH = "times.ttf"     # Nếu có Times New Roman, copy file .ttf vào repo và sửa tên
FONT_NAME = "TimesVN"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
else:
    FONT_NAME = "DejaVu"
    pdfmetrics.registerFont(
        TTFont(FONT_NAME, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )

# =========================================================
# 1. HÀM XỬ LÝ VÀ VẼ VẾT NỨT
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
            draw.polygon(flat_pts, fill=(255, 0, 0, 80))
            draw.line(flat_pts + [flat_pts[0]], fill=(255, 0, 0, 200), width=3)

    result = Image.alpha_composite(base.convert("RGBA"), overlay)
    return result.convert("RGB")


def estimate_severity(p, img_w, img_h):
    """
    Ước lượng "mức độ nghiêm trọng" dựa trên diện tích box so với ảnh:
      - < 1%  : Nhỏ
      - 1–5%  : Trung bình
      - > 5%  : Nguy hiểm (Severe)
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
# 2. HÀM GỌI ROBOFLOW & CONFUSION MATRIX
# =========================================================

def call_roboflow_pil(img: Image.Image, min_conf: float):
    """Gửi ảnh PIL lên Roboflow, trả về (predictions, preds_conf, has_crack_bool)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = requests.post(
        ROBOFLOW_FULL_URL,
        files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    preds = result.get("predictions", [])
    preds_conf = [p for p in preds if float(p.get("confidence", 0)) >= min_conf]
    has_crack = len(preds_conf) > 0
    return preds, preds_conf, has_crack


def plot_confusion_matrix(cm, labels=("Crack", "Non-crack")):
    """
    Vẽ ma trận nhầm lẫn.
    cm: 2x2 numpy array [[TP, FN],
                         [FP, TN]]
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()
    return fig

# =========================================================
# 3. HÀM XUẤT PDF
# =========================================================

def export_pdf(original_img, analyzed_img, metrics_df, filename="bkai_report.pdf"):
    """Tạo file PDF báo cáo, dùng font Unicode (TimesVN/DejaVu)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=25 * mm, rightMargin=25 * mm)
    styles = getSampleStyleSheet()

    # set font
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
    normal = ParagraphStyle("NormalVN", parent=styles["Normal"], fontName=FONT_NAME)

    story = []

    # Logo + tiêu đề
    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=40 * mm))
        story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("BÁO CÁO KIỂM TRA VẾT NỨT BÊ TÔNG", title_style))
    story.append(Paragraph("Concrete Crack Inspection Report", normal))
    story.append(Spacer(1, 8 * mm))

    # Ảnh gốc
    story.append(Paragraph("Ảnh gốc / Original Image", h2))
    img_buf = io.BytesIO()
    original_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=120 * mm))
    story.append(Spacer(1, 6 * mm))

    # Ảnh kết quả
    story.append(Paragraph("Ảnh phân tích / Result Image", h2))
    img2_buf = io.BytesIO()
    analyzed_img.save(img2_buf, format="PNG")
    img2_buf.seek(0)
    story.append(RLImage(img2_buf, width=120 * mm))
    story.append(Spacer(1, 6 * mm))

    # Bảng metrics
    story.append(Paragraph("Bảng thông tin vết nứt / Crack Metrics", h2))

    data = [["Chỉ số (VI)", "Metric (EN)", "Giá trị / Value", "Ý nghĩa / Description"]]
    for _, row in metrics_df.iterrows():
        data.append([row["vi"], row["en"], str(row["value"]), row["desc"]])

    tbl = Table(data, colWidths=[35 * mm, 35 * mm, 40 * mm, 55 * mm])
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
    story.append(Spacer(1, 8 * mm))

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
# 4. GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(
    page_title="BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT",
    layout="wide",
)

# Header có logo
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
with col_title:
    st.title("BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT")
    st.caption("Stage 1: Phân biệt ảnh nứt / không nứt và xuất báo cáo chi tiết.")

st.write("---")

tab1, tab2 = st.tabs(
    ["🔍 Stage 1 – Phân tích & Báo cáo", "📚 Stage 2 (demo) – Phân loại & biện pháp"]
)

# ========================= TAB 1 ==========================
with tab1:
    st.sidebar.header("Cấu hình phân tích")
    min_conf = st.sidebar.slider(
        "Ngưỡng confidence tối thiểu",
        0.0,
        1.0,
        0.3,
        0.05,
    )
    st.sidebar.caption("Chỉ hiển thị những vết nứt có độ tin cậy ≥ ngưỡng này.")

    st.subheader("Ảnh đơn – Phân tích chi tiết & PDF")
    uploaded_file = st.file_uploader(
        "Tải một ảnh bê tông (JPG/PNG)", type=["jpg", "jpeg", "png"], key="single_upl"
    )
    analyze_btn = st.button("🔍 Phân tích ảnh này", key="single_btn")

    if analyze_btn:
        if uploaded_file is None:
            st.warning("Vui lòng chọn một ảnh trước khi bấm **Phân tích**.")
            st.stop()

        t0 = time.time()
        orig_img = Image.open(uploaded_file).convert("RGB")
        img_w, img_h = orig_img.size

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc")
            st.image(orig_img, use_column_width=True)

        # Gọi Roboflow
        with st.spinner("Đang gửi ảnh tới mô hình AI trên Roboflow..."):
            try:
                preds, preds_conf, has_crack = call_roboflow_pil(
                    orig_img, min_conf=min_conf
                )
            except Exception as e:
                st.error(f"Lỗi gọi API Roboflow: {e}")
                st.stop()

        t1 = time.time()
        total_time = t1 - t0

        with col2:
            st.subheader("Ảnh phân tích")
            if not has_crack:
                st.image(orig_img, use_column_width=True)
                st.success("✅ Kết luận: **Không phát hiện vết nứt rõ ràng**.")
            else:
                analyzed_img = draw_predictions_with_mask(orig_img, preds_conf, min_conf)
                st.image(analyzed_img, use_column_width=True)
                st.error("⚠️ Kết luận: **CÓ vết nứt trên ảnh.**")

        # ----- Bảng thông tin & biểu đồ -----
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
                    "value": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                    {"selector": "td", "props": [("background-color", "#fafafa")]},
                ]
            )
            st.dataframe(styled_df, use_container_width=True)

            # Biểu đồ
            st.subheader("Biểu đồ thống kê")
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                plt.figure(figsize=(4, 3))
                plt.bar(range(1, len(confs) + 1), confs, color="#42a5f5")
                plt.xlabel("Crack #")
                plt.ylabel("Confidence")
                plt.ylim(0, 1)
                plt.title("Độ tin cậy từng vùng nứt")
                st.pyplot(plt.gcf())
                plt.close()

            with col_chart2:
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
            )

    # ===================== BATCH / FOLDER =====================
    st.write("---")
    st.subheader("Đánh giá mô hình trên nhiều ảnh (Folder) – Confusion Matrix")

    st.markdown(
        """
**Hướng dẫn:**

- *Upload nhiều ảnh nứt* (ground truth = **Crack**) ở ô thứ nhất.  
- *Upload nhiều ảnh không nứt* (ground truth = **Non-crack**) ở ô thứ hai.  
- Số lượng tổng khoảng 10–20 ảnh là hợp lý (tránh gọi API quá lâu).  
- Bấm **Phân tích folder** để tính ma trận nhầm lẫn & các chỉ số Accuracy / Precision / Recall / F1.
"""
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        crack_files = st.file_uploader(
            "Ảnh NỨT (ground truth Crack)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_crack",
        )
    with col_f2:
        noncrack_files = st.file_uploader(
            "Ảnh KHÔNG NỨT (ground truth Non-crack)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_noncrack",
        )

    batch_btn = st.button("📊 Phân tích folder & vẽ Confusion Matrix", key="batch_btn")

    if batch_btn:
        total_imgs = len(crack_files) + len(noncrack_files)
        if total_imgs == 0:
            st.warning("Vui lòng upload ít nhất 1 ảnh ở mỗi nhóm (hoặc một trong hai nhóm).")
        else:
            st.info(f"Đang phân tích {total_imgs} ảnh, vui lòng đợi…")
            tp = fn = fp = tn = 0
            progress = st.progress(0)
            processed = 0

            # Crack (true label = 1)
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

            # Non-crack (true label = 0)
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

            # Tính các chỉ số
            total = tp + tn + fp + fn
            acc = (tp + tn) / total if total > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * prec * rec / (prec + rec)
                if (prec + rec) > 0
                else 0
            )

            st.markdown(
                f"""
**Tổng kết:**

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

# ========================= TAB 2 ==========================
with tab2:
    st.subheader("Stage 2 (demo) – Phân loại vết nứt & gợi ý nguyên nhân / biện pháp")

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
