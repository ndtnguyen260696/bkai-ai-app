# app.py – Giai đoạn 1: Phân biệt nứt / không nứt + báo cáo chi tiết

import os
import io
import time
import tempfile
import datetime

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from auth import register_user, authenticate_user, init_user_db
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet

# =========================================================
# 1. CẤU HÌNH CẦN SỬA
# =========================================================

# TODO 1: SỬA LẠI CHO ĐÚNG URL ROBOFLOW CỦA BẠN
ROBOFLOW_FULL_URL = (
    "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"
)

# TODO 2: LOGO BKAI – Đặt file logo trong thư mục "logo/"
BKAI_LOGO = "logo.png"

# TODO 3: Tỉ lệ mm / pixel (tạm thời demo, bạn chỉnh theo thực tế)
MM_PER_PIXEL = 0.2  # 1 pixel ≈ 0.2 mm (ví dụ)


# =========================================================
# 2. HÀM VẼ KHUNG VẾT NỨT
# =========================================================
def draw_crack_boxes(image: Image.Image, predictions, min_conf: float = 0.3):
    """Vẽ box + label 'crack 0.95' lên ảnh."""
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

        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        draw.rectangle([x0, y0, x1, y1], outline="#00FF00", width=3)
        label = f"crack {conf:.2f}"
        draw.text((x0 + 3, y0 + 3), label, fill="#00FF00")

    return overlay


# =========================================================
# 3. TÍNH CÁC CHỈ SỐ BẢNG THÔNG TIN VẾT NỨT
# =========================================================
def compute_crack_metrics(
    image_name: str,
    predictions,
    img_w: int,
    img_h: int,
    total_time: float,
    infer_time: float,
):
    """
    Tính các chỉ số:
    - Confidence trung bình
    - Crack Area Ratio (%)
    - Chiều dài / rộng mm
    - Bbox chính (lấy bbox có area lớn nhất)
    - Mức độ nguy hiểm + nhận xét
    """

    if not predictions:
        # Không có vết nứt
        metrics = [
            {
                "Chỉ số (VI)": "Tên ảnh",
                "Metric (EN)": "Image Name",
                "Giá trị / Value": image_name,
                "Ý nghĩa / Description": "File ảnh người dùng tải lên",
            },
            {
                "Chỉ số (VI)": "Nhận xét tổng quan",
                "Metric (EN)": "Summary",
                "Giá trị / Value": "Không phát hiện vết nứt",
                "Ý nghĩa / Description": "Ảnh bê tông không có vết nứt rõ ràng",
            },
        ]
        return pd.DataFrame(metrics), None

    # --------- Tính toán từ predictions ---------
    confs = [float(p.get("confidence", 0)) for p in predictions]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    # mAP ở đây bạn thường biết từ kết quả training → demo = 0.87
    map_val = 0.87

    # Tổng diện tích vùng nứt
    area_img = img_w * img_h
    total_crack_area = 0.0
    main_pred = None
    max_area = -1

    for p in predictions:
        w = float(p.get("width", 0))
        h = float(p.get("height", 0))
        area = w * h
        total_crack_area += area
        if area > max_area:
            max_area = area
            main_pred = p

    crack_area_ratio = (total_crack_area / area_img * 100) if area_img > 0 else 0.0

    # Bbox chính
    if main_pred is not None:
        w_px = float(main_pred.get("width", 0))
        h_px = float(main_pred.get("height", 0))
        x = float(main_pred.get("x", 0))
        y = float(main_pred.get("y", 0))
        bbox = [round(x, 1), round(y, 1), round(w_px, 1), round(h_px, 1)]
    else:
        w_px = h_px = 0
        bbox = [0, 0, 0, 0]

    # Chiều dài / rộng mm (giả định tỉ lệ MM_PER_PIXEL)
    length_px = max(w_px, h_px)
    width_px = min(w_px, h_px)
    crack_length_mm = length_px * MM_PER_PIXEL
    crack_width_mm = width_px * MM_PER_PIXEL

    # Mức độ nguy hiểm theo chiều rộng
    if crack_width_mm < 0.3:
        severity_vi = "Nhẹ"
        severity_en = "Minor"
    elif crack_width_mm < 1.0:
        severity_vi = "Trung bình"
        severity_en = "Moderate"
    else:
        severity_vi = "Nguy hiểm"
        severity_en = "Severe"

    severity_label = f"{severity_vi} ({severity_en})"

    # Nhận xét tổng quan
    if severity_vi == "Nguy hiểm":
        summary = "Vết nứt nguy hiểm, cần kiểm tra và gia cố thêm."
    elif severity_vi == "Trung bình":
        summary = "Vết nứt mức trung bình, nên theo dõi và kiểm tra định kỳ."
    else:
        summary = "Vết nứt nhỏ, ít ảnh hưởng nhưng vẫn cần quan sát."

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Bảng metrics song ngữ
    metrics = [
        {
            "Chỉ số (VI)": "Tên ảnh",
            "Metric (EN)": "Image Name",
            "Giá trị / Value": image_name,
            "Ý nghĩa / Description": "File ảnh người dùng tải lên",
        },
        {
            "Chỉ số (VI)": "Thời gian xử lý",
            "Metric (EN)": "Total Processing Time",
            "Giá trị / Value": f"{total_time:.2f} s",
            "Ý nghĩa / Description": "Tổng thời gian thực hiện toàn bộ quy trình",
        },
        {
            "Chỉ số (VI)": "Tốc độ mô hình AI",
            "Metric (EN)": "Inference Speed",
            "Giá trị / Value": f"{infer_time:.2f} s/image",
            "Ý nghĩa / Description": "Thời gian xử lý một ảnh của mô hình",
        },
        {
            "Chỉ số (VI)": "Confidence trung bình",
            "Metric (EN)": "Confidence",
            "Giá trị / Value": f"{avg_conf:.2f}",
            "Ý nghĩa / Description": "Mức tin cậy trung bình của mô hình",
        },
        {
            "Chỉ số (VI)": "mAP",
            "Metric (EN)": "Mean Average Precision",
            "Giá trị / Value": f"{map_val:.2f}",
            "Ý nghĩa / Description": "Độ chính xác định vị vùng nứt (từ kết quả training)",
        },
        {
            "Chỉ số (VI)": "Phần trăm vùng nứt",
            "Metric (EN)": "Crack Area Ratio",
            "Giá trị / Value": f"{crack_area_ratio:.2f} %",
            "Ý nghĩa / Description": "Diện tích vùng nứt / tổng diện tích ảnh",
        },
        {
            "Chỉ số (VI)": "Chiều dài vết nứt",
            "Metric (EN)": "Crack Length",
            "Giá trị / Value": f"{crack_length_mm:.1f} mm",
            "Ý nghĩa / Description": "Ước tính theo tỉ lệ chuyển đổi pixel → mm",
        },
        {
            "Chỉ số (VI)": "Chiều rộng vết nứt",
            "Metric (EN)": "Crack Width",
            "Giá trị / Value": f"{crack_width_mm:.2f} mm",
            "Ý nghĩa / Description": "Độ rộng lớn nhất của vết nứt",
        },
        {
            "Chỉ số (VI)": "Tọa độ vùng nứt",
            "Metric (EN)": "Crack Bounding Box",
            "Giá trị / Value": str(bbox),
            "Ý nghĩa / Description": "(x, y, w, h) – vị trí vùng nứt chính trên ảnh (pixel)",
        },
        {
            "Chỉ số (VI)": "Mức độ nguy hiểm",
            "Metric (EN)": "Severity Level",
            "Giá trị / Value": severity_label,
            "Ý nghĩa / Description": "Phân cấp theo tiêu chí chiều rộng và vùng ảnh",
        },
        {
            "Chỉ số (VI)": "Thời gian phân tích",
            "Metric (EN)": "Timestamp",
            "Giá trị / Value": timestamp,
            "Ý nghĩa / Description": "Thời điểm thực hiện phân tích",
        },
        {
            "Chỉ số (VI)": "Nhận xét tổng quan",
            "Metric (EN)": "Summary",
            "Giá trị / Value": summary,
            "Ý nghĩa / Description": "Kết luận tự động gợi ý từ mô hình",
        },
    ]

    return pd.DataFrame(metrics), {
        "avg_conf": avg_conf,
        "map": map_val,
        "crack_area_ratio": crack_area_ratio,
        "severity": severity_vi,
    }


# =========================================================
# 4. PHÂN LOẠI VẾT NỨT (GIAI ĐOẠN 2 – DEMO)
# =========================================================
def classify_crack_type(severity: str):
    """
    Demo đơn giản:
    - Dựa theo mức độ nguy hiểm để gợi ý loại nứt, nguyên nhân, biện pháp.
    """
    if severity == "Nguy hiểm":
        crack_type = "Vết nứt kết cấu / Structural crack"
        cause = "Tải trọng vượt thiết kế, lún không đều, cốt thép bị ăn mòn."
        action = "Kiểm định kết cấu, gia cố thép, trám bít bằng vật liệu cường độ cao."
    elif severity == "Trung bình":
        crack_type = "Vết nứt do co ngót / Shrinkage crack"
        cause = "Co ngót bê tông, thay đổi nhiệt độ, độ ẩm trong quá trình đông cứng."
        action = "Theo dõi định kỳ, trám bít bằng vữa/polymer, chống thấm bổ sung."
    else:
        crack_type = "Vết nứt bề mặt / Hairline crack"
        cause = "Lớp vữa hoàn thiện, tác động môi trường, giãn nở nhiệt."
        action = "Làm sạch và sơn/phủ bảo vệ, quan sát thêm nếu phát triển."

    df = pd.DataFrame(
        [
            {
                "Loại vết nứt / Crack Type": crack_type,
                "Nguyên nhân (Cause)": cause,
                "Biện pháp (Recommendation)": action,
            }
        ]
    )
    return df


# =========================================================
# 5. BIỂU ĐỒ + XUẤT PDF
# =========================================================
def create_metrics_chart(metrics_info, out_path):
    """Vẽ biểu đồ bar cho 3 chỉ số: Confidence, mAP, Crack Area Ratio."""
    labels = ["Confidence", "mAP", "CrackAreaRatio(%)"]
    values = [
        metrics_info["avg_conf"],
        metrics_info["map"],
        metrics_info["crack_area_ratio"],
    ]

    plt.figure(figsize=(4, 3))
    plt.bar(labels, values)
    plt.title("Key Metrics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_pdf(
    original_path,
    analyzed_path,
    metrics_df,
    type_df,
    chart_path,
    filename="BKAI_Report.pdf",
):
    """Xuất file PDF đơn giản chứa logo, ảnh, bảng metrics, bảng loại nứt, biểu đồ."""
    tmp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(tmp_dir, filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Logo + tiêu đề
    if os.path.exists(BKAI_LOGO):
        story.append(RLImage(BKAI_LOGO, width=80, height=80))
    story.append(Spacer(1, 10))

    title = "<b>BÁO CÁO KIỂM TRA VẾT NỨT BÊ TÔNG</b>"
    subtitle = "Concrete Crack Inspection Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(subtitle, styles["Heading3"]))
    story.append(Spacer(1, 10))

    # Ảnh
    story.append(Paragraph("<b>Ảnh gốc / Original Image</b>", styles["Heading3"]))
    story.append(RLImage(original_path, width=250, height=180))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Ảnh phân tích / Result Image</b>", styles["Heading3"]))
    story.append(RLImage(analyzed_path, width=250, height=180))
    story.append(Spacer(1, 12))

    # Bảng metrics
    story.append(Paragraph("<b>Bảng thông tin vết nứt / Crack Metrics</b>", styles["Heading3"]))
    data = [list(metrics_df.columns)] + metrics_df.values.tolist()
    tbl = Table(data, colWidths=[90, 90, 90, 160])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Bảng phân loại vết nứt
    if type_df is not None:
        story.append(Paragraph("<b>Phân loại vết nứt (demo)</b>", styles["Heading3"]))
        data2 = [list(type_df.columns)] + type_df.values.tolist()
        tbl2 = Table(data2, colWidths=[120, 160, 160])
        tbl2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        story.append(tbl2)
        story.append(Spacer(1, 12))

    # Biểu đồ
    if os.path.exists(chart_path):
        story.append(Paragraph("<b>Biểu đồ chỉ số / Metrics Chart</b>", styles["Heading3"]))
        story.append(RLImage(chart_path, width=260, height=180))

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "BKAI © Powered by AI for Construction Excellence",
            styles["Normal"],
        )
    )

    doc.build(story)
    return pdf_path


# =========================================================
# 6. GIAO DIỆN STREAMLIT
# =========================================================
st.set_page_config(page_title="BKAI - Crack Inspection (Stage 1)", layout="wide")

# Header
cols_header = st.columns([1, 5])
with cols_header[0]:
    if os.path.exists(BKAI_LOGO):
        st.image(BKAI_LOGO, width=90)
with cols_header[1]:
    st.markdown(
        """
        # BKAI – Concrete Crack Inspection (Stage 1)
        Phân biệt ảnh **nứt / không nứt** và xuất báo cáo chi tiết.
        """,
        unsafe_allow_html=True,
    )

st.write("---")

st.sidebar.header("Cấu hình")
min_conf = st.sidebar.slider(
    "Ngưỡng confidence hiển thị box", 0.0, 1.0, 0.3, 0.05
)

uploaded_file = st.file_uploader(
    "Chọn 1 ảnh bê tông (JPG/PNG)", type=["jpg", "jpeg", "png"]
)
btn_analyze = st.button("🔍 Phân tích ảnh")

# =========================================================
# 7. XỬ LÝ ẢNH
# =========================================================
if btn_analyze:
    if uploaded_file is None:
        st.warning("Vui lòng chọn một ảnh trước.")
        st.stop()

    image_name = uploaded_file.name
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Không đọc được ảnh: {e}")
        st.stop()

    img_w, img_h = image.size
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Ảnh gốc")
        st.image(image, use_column_width=True)

    # ----- Gửi tới Roboflow -----
    total_t0 = time.time()
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    with st.spinner("Đang gửi ảnh tới Roboflow và phân tích..."):
        t0 = time.time()
        try:
            resp = requests.post(
                ROBOFLOW_FULL_URL,
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi khi gọi API Roboflow: {e}")
            st.stop()
        infer_time = time.time() - t0

    total_time = time.time() - total_t0

    if resp.status_code != 200:
        st.error(
            "Roboflow trả lỗi, hãy kiểm tra lại ROBOFLOW_FULL_URL (model_id, version, api_key)."
        )
        st.write(f"Status code: {resp.status_code}")
        st.text(resp.text[:1000])
        st.stop()

    try:
        result = resp.json()
    except Exception as e:
        st.error(f"Không parse được JSON trả về: {e}")
        st.text(resp.text[:2000])
        st.stop()

    predictions = result.get("predictions", [])

    # ----- Phân biệt nứt / không nứt -----
    has_crack = len(predictions) > 0

    with col_right:
        st.subheader("Ảnh phân tích")
        if has_crack:
            annotated = draw_crack_boxes(image, predictions, min_conf=min_conf)
            st.image(annotated, use_column_width=True)
            st.error("⚠️ Kết luận: **CÓ vết nứt** trên ảnh.")
        else:
            st.image(image, use_column_width=True)
            st.success("✅ Kết luận: **KHÔNG phát hiện vết nứt**.")

    st.write("---")
    st.subheader("Bảng thông tin vết nứt")

    metrics_df, metrics_info = compute_crack_metrics(
        image_name=image_name,
        predictions=predictions,
        img_w=img_w,
        img_h=img_h,
        total_time=total_time,
        infer_time=infer_time,
    )

    st.dataframe(metrics_df, use_container_width=True)

    # Nếu không có vết nứt thì không cần giai đoạn 2 + PDF
    if not has_crack or metrics_info is None:
        st.info("Ảnh không có vết nứt, bỏ qua giai đoạn phân loại và PDF.")
        st.stop()

    # =====================================================
    #  Giai đoạn 2 (demo): phân loại vết nứt
    # =====================================================
    st.subheader("Giai đoạn 2 – Phân loại vết nứt (demo)")
    crack_type_df = classify_crack_type(metrics_info["severity"])
    st.table(crack_type_df)

    # =====================================================
    #  Biểu đồ từ bảng metrics
    # =====================================================
    st.subheader("Biểu đồ tổng hợp từ các chỉ số")

    tmp_dir = tempfile.gettempdir()
    chart_path = os.path.join(tmp_dir, "bkai_metrics_chart.png")
    create_metrics_chart(metrics_info, chart_path)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.image(chart_path, caption="Key Metrics (Confidence, mAP, CrackAreaRatio)", use_column_width=True)

    with col_c2:
        # Biểu đồ đơn giản: phần trăm vùng nứt vs phần còn lại
        fig, ax = plt.subplots(figsize=(4, 3))
        crack_ratio = metrics_info["crack_area_ratio"]
        ax.pie(
            [crack_ratio, max(0, 100 - crack_ratio)],
            labels=["Crack Area", "Intact Area"],
            autopct="%1.1f%%",
        )
        ax.set_title("Crack vs Intact Area")
        st.pyplot(fig)

    # =====================================================
    #  Xuất báo cáo PDF
    # =====================================================
    st.subheader("Báo cáo PDF")

    # Lưu tạm ảnh
    original_path = os.path.join(tmp_dir, "bkai_original.jpg")
    analyzed_path = os.path.join(tmp_dir, "bkai_analyzed.jpg")
    image.save(original_path, format="JPEG")
    annotated.save(analyzed_path, format="JPEG")

    pdf_filename = f"BKAI_Crack_Report_{image_name}.pdf"
    pdf_path = export_pdf(
        original_path=original_path,
        analyzed_path=analyzed_path,
        metrics_df=metrics_df,
        type_df=crack_type_df,
        chart_path=chart_path,
        filename=pdf_filename,
    )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        "📄 Tải báo cáo PDF cho ảnh này",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        key="download_pdf",  # key riêng → không lỗi DuplicateElementId
    )


