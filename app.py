import streamlit as st
import requests
from PIL import Image, ImageDraw
import io, datetime, tempfile, os
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# =========================================================
# 1. CẤU HÌNH ROBOFLOW
# =========================================================
ROBOFLOW_FULL_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"

BKAI_LOGO = "bkai_logo.png"  # Đặt logo BKAI cùng thư mục app.py

# =========================================================
# 2. HÀM XỬ LÝ DỮ LIỆU
# =========================================================
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

def draw_predictions(image: Image.Image, predictions, min_conf: float = 0.0) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for p in predictions:
        conf = float(p.get("confidence", 0))
        if conf < min_conf:
            continue
        x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
        if None in (x, y, w, h): continue
        x0, y0, x1, y1 = x - w/2, y - h/2, x + w/2, y + h/2
        draw.rectangle([x0, y0, x1, y1], outline="#A020F0", width=3)
        label = f"{p.get('class','crack')} {conf:.2f}"
        draw.text((x0 + 3, y0 + 3), label, fill="#A020F0")
        pts = p.get("points")
        flat_pts = extract_poly_points(pts) if pts is not None else []
        if len(flat_pts) >= 2:
            draw.line(flat_pts, fill="#A020F0", width=2)
    return overlay

def estimate_severity(p, img_w, img_h):
    w, h = float(p.get("width", 0)), float(p.get("height", 0))
    if img_w <= 0 or img_h <= 0: return "Không xác định"
    ratio = (w * h) / (img_w * img_h)
    if ratio < 0.01: return "Nhỏ"
    elif ratio < 0.05: return "Trung bình"
    else: return "Lớn"

# =========================================================
# 3. XUẤT PDF BÁO CÁO
# =========================================================
def export_pdf(original_path, analyzed_path, df, chart_path, filename):
    pdf_path = os.path.join(tempfile.gettempdir(), f"BKAI_Report_{filename}.pdf")
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    story = []

    if os.path.exists(BKAI_LOGO):
        story.append(RLImage(BKAI_LOGO, width=100, height=100))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>BKAI – Concrete Crack Inspection Report</b>", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Original Image</b>", styles["Heading3"]))
    story.append(RLImage(original_path, width=250, height=150))
    story.append(Spacer(1, 5))

    story.append(Paragraph("<b>Analyzed Image</b>", styles["Heading3"]))
    story.append(RLImage(analyzed_path, width=250, height=150))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Crack Information Summary</b>", styles["Heading3"]))
    table_data = [df.columns.tolist()] + df.values.tolist()
    tbl = Table(table_data)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Crack Charts</b>", styles["Heading3"]))
    story.append(RLImage(chart_path, width=350, height=200))
    story.append(Spacer(1, 10))

    story.append(Paragraph("BKAI © 2025 – Powered by AI for Construction Excellence", styles["Normal"]))
    doc.build(story)
    return pdf_path

# =========================================================
# 4. GIAO DIỆN STREAMLIT
# =========================================================
st.set_page_config(page_title="BKAI - Crack Analysis", layout="wide")
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(BKAI_LOGO):
        st.image(BKAI_LOGO, width=120)
with col_title:
    st.markdown("<h1 style='text-align:center;'>BKAI – Concrete Crack Detection and Analysis</h1>", unsafe_allow_html=True)
st.divider()

st.sidebar.header("⚙️ Tùy chỉnh phân tích")
min_conf = st.sidebar.slider("Ngưỡng Confidence tối thiểu", 0.0, 1.0, 0.3, 0.05)
st.sidebar.caption("Chỉ hiển thị vết nứt có độ tin cậy cao hơn ngưỡng này.")

uploaded_file = st.file_uploader("📂 Chọn ảnh bê tông (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = image.size
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    with st.spinner("Đang gửi ảnh đến mô hình Roboflow..."):
        try:
            resp = requests.post(ROBOFLOW_FULL_URL, files={"file": ("image.jpg", img_bytes, "image/jpeg")})
            result = resp.json()
        except Exception as e:
            st.error(f"Lỗi khi gọi API: {e}")
            st.stop()

    preds = [p for p in result.get("predictions", []) if float(p.get("confidence", 0)) >= min_conf]
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Ảnh gốc / Original", use_column_width=True)
    with col2:
        annotated = draw_predictions(image, preds, min_conf)
        st.image(annotated, caption="Ảnh phân tích / Analyzed", use_column_width=True)

    if len(preds) == 0:
        st.success("✅ Không phát hiện vết nứt rõ ràng.")
    else:
        st.error("⚠️ Có vết nứt được phát hiện!")

    # Bảng thông số
    rows = []
    for i, p in enumerate(preds, start=1):
        sev = estimate_severity(p, img_w, img_h)
        rows.append({
            "Crack #": i,
            "Confidence": round(p["confidence"], 3),
            "Width(px)": round(p["width"], 1),
            "Height(px)": round(p["height"], 1),
            "Severity": sev,
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Biểu đồ
        confs = [r["Confidence"] for r in rows]
        widths = [r["Width(px)"] for r in rows]
        heights = [r["Height(px)"] for r in rows]

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].bar(range(len(confs)), confs, color="#0ea5e9")
        axs[0].set_title("Confidence per Crack")
        axs[1].pie([len(rows), 10 - len(rows)], labels=["Crack", "No Crack"], autopct="%1.0f%%", colors=["#ef4444", "#22c55e"])
        axs[1].set_title("Crack Ratio")
        axs[2].scatter(widths, heights, c=confs, cmap="plasma", s=80)
        axs[2].set_title("Crack Size Distribution")
        plt.tight_layout()

        tmp_chart = os.path.join(tempfile.gettempdir(), "chart.png")
        fig.savefig(tmp_chart)
        st.pyplot(fig)

        # Xuất PDF
        orig_path = os.path.join(tempfile.gettempdir(), "orig.png")
        ann_path = os.path.join(tempfile.gettempdir(), "ann.png")
        image.save(orig_path)
        annotated.save(ann_path)
        pdf_path = export_pdf(orig_path, ann_path, df, tmp_chart, uploaded_file.name)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Tải báo cáo PDF", f.read(), file_name=f"BKAI_Report_{uploaded_file.name}.pdf", mime="application/pdf")

else:
    st.info("⬆️ Hãy tải một ảnh để bắt đầu phân tích.")
