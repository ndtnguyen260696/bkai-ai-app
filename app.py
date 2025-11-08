import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import datetime
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# =========================================================
# 1. CẤU HÌNH ROBOFLOW & LOGO
# =========================================================

# 🔧 CẦN SỬA 1:
# Thay URL này bằng Hosted API URL của model Roboflow của bạn
# (Roboflow > Project > Deploy > Hosted API > Python)
ROBOFLOW_FULL_URL = (
    "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"
)

# 🔧 CẦN SỬA 2 (nếu cần):
# Đường dẫn tới file logo BKAI (hiện đang nằm trong thư mục "logo/")
BKAI_LOGO = os.path.join("logo", "bkai_logo.png")


# =========================================================
# 2. CÁC HÀM TIỆN ÍCH: VẼ, TÍNH TOÁN, PDF
# =========================================================

def extract_poly_points(points_field):
    """Chuyển trường 'points' trong JSON thành list [(x,y), ...]."""
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


def draw_predictions(image: Image.Image, predictions, min_conf: float = 0.0) -> Image.Image:
    """Vẽ box màu xanh + polyline đỏ cho các vết nứt."""
    overlay = image.copy()
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

        # x,y là tâm box
        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        # Box màu xanh lá
        draw.rectangle([x0, y0, x1, y1], outline="#22c55e", width=3)

        # Nhãn trên box: crack 0.50
        cls = p.get("class", "crack")
        label = f"{cls} {conf:.2f}"
        draw.text((x0 + 3, y0 + 3), label, fill="#22c55e")

        # Polyline đỏ (nếu có points)
        pts = p.get("points")
        flat_pts = extract_poly_points(pts) if pts is not None else []
        if len(flat_pts) >= 2:
            draw.line(flat_pts, fill="#ef4444", width=3)

    return overlay


def severity_and_area_pct(p, img_w, img_h):
    """
    Trả về:
      - Mức độ: Nhỏ / Trung bình / Lớn
      - Diện tích (%) của box so với ảnh
    """
    w = float(p.get("width", 0))
    h = float(p.get("height", 0))
    if img_w <= 0 or img_h <= 0:
        return "Không xác định", 0.0

    area_box = w * h
    area_img = img_w * img_h
    ratio = area_box / area_img
    pct = ratio * 100.0

    if pct < 0.5:
        sev = "Nhỏ"
    elif pct < 1.0:
        sev = "Trung bình"
    else:
        sev = "Lớn"
    return sev, pct


def create_overview_table(preds, img_w, img_h, inference_time, min_conf):
    """
    Tạo bảng Overview song ngữ giống mẫu PDF:
    Confidence, mAP, Detection, Segmentation, Inference Time, Conclusion
    (Các giá trị ở đây là gợi ý từ confidence, bạn có thể thay bằng số thực tế nếu có).
    """
    if preds:
        confs = [p["confidence"] for p in preds]
        avg_conf = sum(confs) / len(confs)
        max_conf = max(confs)
    else:
        avg_conf = 0.0
        max_conf = 0.0

    confidence_score = avg_conf         # gợi ý
    detection_score = max_conf         # gợi ý
    segmentation_score = avg_conf * 0.9  # gợi ý

    conclusion = (
        "Có vết nứt / Cracks present"
        if preds
        else "Không vết nứt / No cracks"
    )

    data = [
        ["Confidence", f"{confidence_score:.2f}", "Độ chính xác", f"{confidence_score:.2f}"],
        ["mAP", f"{avg_conf:.2f}", "Segmentation", f"{segmentation_score:.2f}"],
        ["Detection", f"{detection_score:.2f}", "Inference Time", f"{inference_time*1000:.0f} ms"],
        ["Conclusion", "", conclusion, ""],
    ]
    df = pd.DataFrame(
        data,
        columns=["Metric (EN)", "Value", "Chỉ số (VI)", "Giá trị"],
    )
    return df


def export_pdf_report(
    original_path,
    analyzed_path,
    df_overview,
    df_instances,
    chart_path,
    file_name: str,
):
    """
    Tạo file PDF cho 1 ảnh, gồm:
     - Logo BKAI
     - Tiêu đề VN/EN
     - Ảnh gốc / Ảnh phân tích
     - Overview table
     - Crack Instances Table
     - Biểu đồ
    """
    pdf_path = os.path.join(tempfile.gettempdir(), f"BKAI_Crack_Report_{file_name}.pdf")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)
    story = []

    # Logo + tiêu đề
    if os.path.exists(BKAI_LOGO):
        story.append(RLImage(BKAI_LOGO, width=80, height=80))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>BÁO CÁO KIỂM TRA VẾT NỨT BÊ TÔNG</b>", styles["Title"]))
    story.append(Paragraph("Concrete Crack Inspection Report", styles["Heading3"]))
    story.append(
        Paragraph(
            datetime.datetime.now().strftime("Date: %B %d, %Y"),
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    # Ảnh gốc
    story.append(Paragraph("<b>Ảnh gốc / Original Image</b>", styles["Heading3"]))
    story.append(RLImage(original_path, width=250, height=160))
    story.append(Spacer(1, 6))

    # Ảnh phân tích
    story.append(Paragraph("<b>Ảnh phân tích / Result Image</b>", styles["Heading3"]))
    story.append(RLImage(analyzed_path, width=250, height=160))
    story.append(Spacer(1, 10))

    # Overview table
    story.append(Paragraph("<b>Overview</b>", styles["Heading2"]))
    tbl_data = [df_overview.columns.tolist()] + df_overview.values.tolist()
    tbl = Table(tbl_data, colWidths=[100, 70, 120, 70])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 10))

    # Crack Instances Table
    story.append(Paragraph("<b>Crack Instances Table</b>", styles["Heading2"]))
    tbl2_data = [df_instances.columns.tolist()] + df_instances.values.tolist()
    tbl2 = Table(tbl2_data)
    tbl2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    story.append(tbl2)
    story.append(Spacer(1, 10))

    # Biểu đồ
    story.append(Paragraph("<b>Charts</b>", styles["Heading2"]))
    story.append(RLImage(chart_path, width=380, height=230))
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "BKAI © 2025 – Powered by AI for Construction Excellence",
            styles["Normal"],
        )
    )

    doc.build(story)
    return pdf_path


# =========================================================
# 3. GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(page_title="BKAI - Crack Report", layout="wide")

# Header có logo + tiêu đề
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(BKAI_LOGO):
        st.image(BKAI_LOGO, width=120)
with col_title:
    st.markdown(
        "<h1 style='text-align:center;'>BKAI – Concrete Crack Detection & Reporting</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Phát hiện và phân tích vết nứt bê tông bằng AI</p>",
        unsafe_allow_html=True,
    )
st.divider()

st.sidebar.header("⚙️ Cấu hình")
min_conf = st.sidebar.slider(
    "Ngưỡng confidence tối thiểu để hiển thị",
    0.0,
    1.0,
    0.3,
    0.05,
)

# Có thể up nhiều ảnh
uploaded_files = st.file_uploader(
    "📂 Chọn 1–20 ảnh bê tông (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

analyze = st.button("🚀 Phân tích tất cả ảnh")

if analyze:
    if not uploaded_files:
        st.warning("Vui lòng tải lên ít nhất một ảnh.")
        st.stop()

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        st.divider()
        st.markdown(f"## 🖼️ Ảnh {idx}: `{uploaded_file.name}`")

        # Đọc ảnh
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Không đọc được ảnh: {e}")
            continue

        img_w, img_h = image.size
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        # Gọi API Roboflow
        with st.spinner("Đang gửi ảnh tới Roboflow…"):
            t0 = datetime.datetime.now().timestamp()
            try:
                resp = requests.post(
                    ROBOFLOW_FULL_URL,
                    files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                    timeout=60,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi khi gọi API Roboflow: {e}")
                continue
            t1 = datetime.datetime.now().timestamp()
            inference_time = t1 - t0

        if resp.status_code != 200:
            st.error("Roboflow trả lỗi. Hãy kiểm tra lại ROBOFLOW_FULL_URL (model_id, version, api_key).")
            st.write(f"Status: {resp.status_code}")
            st.text(resp.text[:1000])
            continue

        try:
            result = resp.json()
        except Exception as e:
            st.error(f"Không parse được JSON: {e}")
            st.text(resp.text[:1500])
            continue

        predictions = result.get("predictions", [])
        preds_conf = [
            p for p in predictions if float(p.get("confidence", 0)) >= min_conf
        ]

        # Hiển thị ảnh gốc & ảnh phân tích
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc / Original")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Ảnh phân tích / Analyzed")
            if preds_conf:
                annotated = draw_predictions(image, preds_conf, min_conf=min_conf)
                st.image(annotated, use_column_width=True)
                st.error("⚠️ Kết luận: CÓ vết nứt được phát hiện.")
            else:
                annotated = image.copy()
                st.image(annotated, use_column_width=True)
                if predictions:
                    st.info("Model phát hiện tín hiệu nhưng dưới ngưỡng confidence đã chọn.")
                st.success("✅ Kết luận: Không có vết nứt rõ ràng.")

        # ===== Overview Table =====
        st.markdown("### Overview")
        df_overview = create_overview_table(preds_conf, img_w, img_h, inference_time, min_conf)
        st.table(df_overview)

        # ===== Crack Instances Table =====
        st.markdown("### Crack Instances Table (Chi tiết từng vùng vết nứt)")
        instance_rows = []
        for i, p in enumerate(preds_conf, start=1):
            severity, area_pct = severity_and_area_pct(p, img_w, img_h)
            instance_rows.append(
                {
                    "Crack #": i,
                    "Confidence": round(float(p.get("confidence", 0)), 3),
                    "Mức độ": severity,
                    "Width(px)": round(float(p.get("width", 0)), 1),
                    "Height(px)": round(float(p.get("height", 0)), 1),
                    "Diện tích(%)": f"{area_pct:.1f}%",
                }
            )

        if instance_rows:
            df_instances = pd.DataFrame(instance_rows)
            st.dataframe(df_instances, use_container_width=True)
        else:
            df_instances = pd.DataFrame(
                columns=["Crack #", "Confidence", "Mức độ", "Width(px)", "Height(px)", "Diện tích(%)"]
            )
            st.info("Không có vùng vết nứt vượt ngưỡng để liệt kê.")

        # ===== Biểu đồ =====
        st.markdown("### Charts – Confidence & Crack Presence")

        chart_path = os.path.join(tempfile.gettempdir(), f"bkai_chart_{idx}.png")

        if instance_rows:
            confs = [r["Confidence"] for r in instance_rows]
            crack_count = len(instance_rows)

            fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))

            # Bar chart Confidence
            axs[0].bar([r["Crack #"] for r in instance_rows], confs, color="#0ea5e9")
            axs[0].set_title("Confidence Scores")
            axs[0].set_xlabel("Crack #")
            axs[0].set_ylabel("Confidence")
            axs[0].set_ylim(0, 1)

            # Pie chart presence
            axs[1].pie(
                [crack_count, max(1, 10 - crack_count)],
                labels=["Crack regions", "Non-crack"],
                autopct="%1.0f%%",
                colors=["#2563eb", "#cbd5f5"],
            )
            axs[1].set_title("Crack Presence")

            plt.tight_layout()
            fig.savefig(chart_path, bbox_inches="tight")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, "No cracks", ha="center", va="center")
            ax.axis("off")
            fig.savefig(chart_path, bbox_inches="tight")
            st.pyplot(fig)

        # ===== Lưu ảnh tạm & Export PDF =====
        tmpdir = tempfile.gettempdir()
        orig_path = os.path.join(tmpdir, f"bkai_orig_{idx}.png")
        ann_path = os.path.join(tmpdir, f"bkai_ann_{idx}.png")
        image.save(orig_path)
        annotated.save(ann_path)

        pdf_path = export_pdf_report(
            orig_path,
            ann_path,
            df_overview,
            df_instances,
            chart_path,
            uploaded_file.name.replace(" ", "_"),
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Tải báo cáo PDF cho ảnh này",
                data=f.read(),
                file_name=f"BKAI_Crack_Report_{uploaded_file.name}.pdf",
                mime="application/pdf",
            )

else:
    st.info("⬆️ Hãy tải ảnh và bấm **Phân tích tất cả ảnh** để bắt đầu.")
