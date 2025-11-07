import os
import io
import time
import datetime
import tempfile

import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# =========================================================
# 1. CẤU HÌNH ROBOFLOW
# =========================================================
# Copy nguyên link Hosted API từ Roboflow (Deploy -> Hosted API -> Python)
# Ví dụ:
#   "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=xxxx"
ROBOFLOW_FULL_URL ="https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"


# =========================================================
# 2. LOGO & BRAND BKAI
# =========================================================
BKAI_LOGO = "bkai_logo.png"
BKAI_SITE = "https://bkai.b12sites.com/index"


def show_logo(size: int = 100):
    """Hiển thị logo BKAI hoặc link website."""
    if os.path.exists(BKAI_LOGO):
        st.image(BKAI_LOGO, width=size)
    else:
        st.markdown(f"[🌐 BKAI Website]({BKAI_SITE})")


# =========================================================
# 3. HÀM HỖ TRỢ XỬ LÝ ẢNH & PREDICTIONS
# =========================================================
def resize_for_speed(img: Image.Image, max_side: int):
    """Resize ảnh nhưng giữ tỉ lệ, cạnh dài nhất = max_side (nếu > max_side)."""
    w, h = img.size
    max_dim = max(w, h)
    if max_dim <= max_side:
        return img, 1.0
    scale = max_side / max_dim
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size), scale


def extract_points(points_field):
    """Chuyển trường 'points' từ JSON của Roboflow thành list [(x,y), ...]."""
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


def draw_predictions(image: Image.Image, predictions, min_conf: float):
    """
    Vẽ:
      - Mask đỏ trong suốt (instance segmentation) nếu có points
      - Box xanh
      - Label 'crack 0.xx' như Ultralytics
    """
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base)
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)

    blue = (0, 180, 255)
    red_fill = (255, 0, 0, 80)
    red_outline = (255, 0, 0, 180)

    for p in predictions:
        conf = float(p.get("confidence", 0))
        if conf < min_conf:
            continue

        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2

        # Mask từ polygon
        pts = extract_points(p.get("points", []))
        if len(pts) >= 3:
            mask_draw.polygon(pts, fill=red_fill, outline=red_outline)
        elif len(pts) >= 2:
            mask_draw.line(pts, fill=red_outline, width=3)

        # Box
        draw.rectangle([x0, y0, x1, y1], outline=blue, width=3)

        # Label
        label = f"{p.get('class', 'crack')} {conf:.2f}"
        label_bg_w = 90
        label_bg_h = 18
        draw.rectangle([x0, y0 - label_bg_h, x0 + label_bg_w, y0], fill=blue)
        draw.text((x0 + 3, y0 - label_bg_h + 2), label, fill="white")

    return Image.alpha_composite(base, mask_layer).convert("RGB")


def estimate_severity(p, img_w, img_h):
    """Ước lượng mức độ nghiêm trọng từ diện tích box / diện tích ảnh."""
    w, h = float(p["width"]), float(p["height"])
    ratio = (w * h) / (img_w * img_h)
    if ratio < 0.01:
        return "Nhỏ / Small"
    elif ratio < 0.05:
        return "Trung bình / Medium"
    else:
        return "Lớn / Large"


# =========================================================
# 4. HÀM XUẤT PDF BÁO CÁO
# =========================================================
def export_report_pdf(
    pdf_path,
    original_path,
    annotated_path,
    df_summary,
    chart_path,
    title="BKAI – Báo cáo kiểm tra vết nứt bê tông / Concrete Crack Inspection Report",
):
    """
    Tạo file PDF báo cáo:
      - Logo, tiêu đề
      - Ảnh gốc + ảnh kết quả
      - Bảng tổng quan song ngữ
      - Biểu đồ confidence (bar/pie)
    """
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Tiêu đề
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # Thời gian
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Thời gian / Generated at: {now_str}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Ảnh gốc + ảnh kết quả
    story.append(Paragraph("<b>Ảnh gốc / Original Image</b>", styles["Heading3"]))
    story.append(RLImage(original_path, width=260, height=180))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Ảnh kết quả / Analyzed Image</b>", styles["Heading3"]))
    story.append(RLImage(annotated_path, width=260, height=180))
    story.append(Spacer(1, 18))

    # Bảng tổng quan
    story.append(
        Paragraph("<b>Báo cáo tổng quan / Summary Analysis</b>", styles["Heading3"])
    )
    data = [df_summary.columns.tolist()] + df_summary.values.tolist()
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 18))

    # Biểu đồ
    story.append(
        Paragraph(
            "<b>Biểu đồ độ tin cậy / Confidence Charts</b>", styles["Heading3"]
        )
    )
    story.append(RLImage(chart_path, width=400, height=230))
    story.append(Spacer(1, 12))

    # Footer
    story.append(
        Paragraph(
            "BKAI © 2025 – Powered by AI for Construction Excellence",
            styles["Normal"],
        )
    )

    doc.build(story)
    return pdf_path


# =========================================================
# 5. GIAO DIỆN STREAMLIT
# =========================================================
st.set_page_config(page_title="BKAI Crack Inspection Pro", layout="wide")

# Dark theme nhẹ
st.markdown(
    """
<style>
body { background-color: #020617; color: #e5e7eb; }
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
h1, h2, h3 { color: #0ea5e9; }
table, th, td { color: #e5e7eb !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    show_logo(130)
    st.markdown("### ⚙️ Cấu hình / Settings")
    min_conf = st.slider(
        "Ngưỡng confidence / Confidence threshold",
        0.0,
        1.0,
        0.3,
        0.05,
    )
    max_side = st.slider(
        "Kích thước ảnh tối đa / Max image size (px)",
        400,
        1600,
        900,
        100,
    )
    st.caption(
        "Ảnh lớn sẽ được thu nhỏ về kích thước này để tăng tốc xử lý.\n"
        "Large images will be resized to speed up inference."
    )

# Header
col_logo, col_title = st.columns([1, 4])
with col_logo:
    show_logo(80)
with col_title:
    st.title(
        "🧠 BKAI – Báo cáo kiểm tra vết nứt bê tông / Concrete Crack Inspection Report"
    )

st.markdown(
    """
Ứng dụng sử dụng **mô hình AI của BKAI + Roboflow** để:
- 🟥 Tô **vùng nứt** đỏ trong suốt (Instance Segmentation)
- 🟦 Khoanh vùng bằng **box xanh** + label `crack 0.xx`
- 📊 Tạo **bảng tổng quan** + **bảng chi tiết** vết nứt (song ngữ)
- 📈 Vẽ **nhiều dạng biểu đồ** (bar + pie) về độ tin cậy
- 📄 Xuất **PDF báo cáo** đầy đủ cho từng ảnh
"""
)

# Upload
uploaded_files = st.file_uploader(
    "📂 Chọn 1 hoặc nhiều ảnh bê tông (JPG/PNG) / Select one or multiple images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# =========================================================
# 6. XỬ LÝ TỪNG ẢNH
# =========================================================
if uploaded_files:
    for idx, up in enumerate(uploaded_files, start=1):
        st.write("---")
        st.markdown(f"## 🖼️ Ảnh {idx}: `{up.name}`")

        # Đọc & resize ảnh
        image = Image.open(up).convert("RGB")
        image, scale = resize_for_speed(image, max_side)
        img_w, img_h = image.size

        # Encode ảnh
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        # Gọi API
        with st.spinner("⏳ Đang phân tích ảnh với AI BKAI... / Analyzing image..."):
            t0 = time.time()
            try:
                resp = requests.post(
                    ROBOFLOW_FULL_URL,
                    files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                    timeout=60,
                )
            except Exception as e:
                st.error(f"Lỗi khi gọi API Roboflow: {e}")
                continue
            latency = time.time() - t0

        if resp.status_code != 200:
            st.error(f"Roboflow API trả lỗi {resp.status_code}")
            st.text(resp.text[:500])
            continue

        try:
            result = resp.json()
        except Exception as e:
            st.error(f"Không parse được JSON trả về: {e}")
            st.text(resp.text[:500])
            continue

        preds = result.get("predictions", [])
        preds_conf = [p for p in preds if p.get("confidence", 0) >= min_conf]

        # Hiển thị ảnh
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc / Original")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Ảnh kết quả / Analyzed")
            if preds_conf:
                annotated = draw_predictions(image, preds_conf, min_conf)
                st.image(annotated, use_column_width=True)
                st.error("⚠️ Có vết nứt / Crack detected")
            else:
                annotated = image.copy()
                st.image(annotated, use_column_width=True)
                st.success("✅ Không phát hiện vết nứt đáng kể / No significant crack")

        # Nếu không có bất kỳ prediction nào, bỏ qua báo cáo chi tiết
        if not preds:
            st.info("Model không phát hiện vùng nào / No predictions from model.")
            continue

        # =====================================================
        # 6.1 BÁO CÁO TỔNG QUAN (SONG NGỮ)
        # =====================================================
        confs_all = [float(p["confidence"]) for p in preds]
        avg_conf = sum(confs_all) / len(confs_all)
        max_conf = max(confs_all)
        min_conf_pred = min(confs_all)

        area_crack = sum(float(p["width"]) * float(p["height"]) for p in preds_conf)
        coverage = (area_crack / (img_w * img_h)) * 100 if img_w * img_h > 0 else 0.0

        conclusion = (
            "Có vết nứt / Crack detected"
            if preds_conf
            else "Không có vết nứt rõ ràng / No clear crack"
        )

        df_summary = pd.DataFrame(
            [
                [
                    "Kết luận / Conclusion",
                    conclusion,
                    "Đánh giá tổng thể từ các vùng phát hiện / Overall assessment",
                ],
                [
                    "Số vùng phát hiện / Total regions",
                    len(preds),
                    "Tất cả vùng model phát hiện / All predictions",
                ],
                [
                    "Số vùng đạt ngưỡng / Regions ≥ threshold",
                    len(preds_conf),
                    f"Confidence ≥ {min_conf:.2f}",
                ],
                [
                    "Độ tin cậy TB / Avg confidence",
                    f"{avg_conf:.3f}",
                    "Trung bình trên tất cả vùng / Average over all regions",
                ],
                [
                    "Độ tin cậy cao nhất / Max confidence",
                    f"{max_conf:.3f}",
                    "",
                ],
                [
                    "Độ tin cậy thấp nhất / Min confidence",
                    f"{min_conf_pred:.3f}",
                    "",
                ],
                [
                    "Độ phủ vết nứt / Surface coverage",
                    f"{coverage:.2f} %",
                    "Tỉ lệ diện tích box so với ảnh / Crack area ratio",
                ],
                [
                    "Thời gian xử lý / Processing time",
                    f"{latency:.2f} s",
                    "Bao gồm upload + inference / Including upload + inference",
                ],
                [
                    "Kích thước ảnh xử lý / Image size",
                    f"{img_w} × {img_h} px",
                    "Sau khi resize / After resizing",
                ],
                [
                    "Ngưỡng confidence / Threshold",
                    f"{min_conf:.2f}",
                    "",
                ],
                [
                    "F1-score",
                    "N/A",
                    "Cần tập test có ground truth / Requires labeled test set",
                ],
                [
                    "mAP",
                    "N/A",
                    "Không tính từ 1 ảnh / Not computed per single image",
                ],
            ],
            columns=["Chỉ số / Indicator", "Giá trị / Value", "Ghi chú / Notes"],
        )

        st.subheader("📊 Báo cáo tổng quan / Summary report")
        st.table(df_summary)

        # =====================================================
        # 6.2 CHI TIẾT VẾT NỨT
        # =====================================================
        details = []
        for i_p, p in enumerate(preds_conf, start=1):
            details.append(
                {
                    "Crack #": i_p,
                    "Confidence": round(float(p["confidence"]), 3),
                    "Mức độ / Severity": estimate_severity(p, img_w, img_h),
                    "Width(px)": round(float(p["width"]), 1),
                    "Height(px)": round(float(p["height"]), 1),
                }
            )

        st.subheader("🔎 Chi tiết vết nứt / Crack details")
        if details:
            df_details = pd.DataFrame(details)
            st.dataframe(df_details, use_container_width=True)
        else:
            st.write(
                "Không có vùng nào vượt ngưỡng hiển thị / "
                "No region above threshold to show details."
            )

        # =====================================================
        # 6.3 BIỂU ĐỒ KẾT QUẢ
        # =====================================================
        st.subheader("📈 Biểu đồ kết quả / Result charts")

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Bar chart confidence
        axs[0].bar(
            list(range(1, len(confs_all) + 1)),
            confs_all,
            color="#0ea5e9",
        )
        axs[0].set_title("Confidence từng vùng / per region")
        axs[0].set_xlabel("Region #")
        axs[0].set_ylabel("Confidence")
        axs[0].set_ylim(0, 1)

        # Pie chart: trên ngưỡng vs dưới ngưỡng
        above = len(preds_conf)
        below = len(preds) - above
        axs[1].pie(
            [above, below],
            labels=["≥ threshold", "< threshold"],
            autopct="%1.0f%%",
            colors=["#22c55e", "#64748b"],
        )
        axs[1].set_title("Phân bố vùng nứt / Crack distribution")

        st.pyplot(fig)

        # =====================================================
        # 6.4 TẠO PDF & NÚT DOWNLOAD
        # =====================================================
        tmp_dir = tempfile.gettempdir()
        orig_path = os.path.join(tmp_dir, f"bkai_orig_{idx}.png")
        ann_path = os.path.join(tmp_dir, f"bkai_ann_{idx}.png")
        chart_path = os.path.join(tmp_dir, f"bkai_chart_{idx}.png")
        pdf_path = os.path.join(tmp_dir, f"BKAI_Crack_Report_{idx}.pdf")

        image.save(orig_path)
        annotated.save(ann_path)
        fig.savefig(chart_path, bbox_inches="tight")
        plt.close(fig)

        export_report_pdf(pdf_path, orig_path, ann_path, df_summary, chart_path)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Tải báo cáo PDF / Download PDF report",
                data=f.read(),
                file_name=f"BKAI_Crack_Report_{idx}.pdf",
                mime="application/pdf",
            )

else:
    st.info("⬆️ Vui lòng tải lên ít nhất 1 ảnh để bắt đầu phân tích / Upload images to start.")
