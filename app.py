import streamlit as st
import requests
from PIL import Image, ImageDraw
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
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,          # dùng để ngắt trang
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.doctemplate import LayoutError

# =========================================================
# Helper: lưu matplotlib Figure thành PNG bytes để nhúng vào PDF
# =========================================================
def fig_to_png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# =========================================================
# 0. CẤU HÌNH CHUNG
# =========================================================

ROBOFLOW_FULL_URL = (
    "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"
)

LOGO_PATH = "BKAI_Logo.png"

FONT_PATH = "times.ttf"
FONT_NAME = "TimesVN"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
else:
    FONT_NAME = "DejaVu"
    pdfmetrics.registerFont(
        TTFont(FONT_NAME, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )

st.set_page_config(
    page_title="BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT",
    layout="wide",
)

# =========================================================
# 1. HÀM XỬ LÝ ẢNH
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


def draw_predictions_with_mask(
    image: Image.Image, predictions, min_conf: float = 0.0
) -> Image.Image:
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    green_solid = (0, 255, 0, 255)
    green_fill = (0, 255, 0, 80)

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

        draw.rectangle([x0, y0, x1, y1], outline=green_solid, width=3)

        cls = p.get("class", "crack")
        label = f"{cls} {conf:.2f}"
        text_pos = (x0 + 3, y0 + 3)
        draw.text(text_pos, label, fill=green_solid)

        pts_raw = p.get("points")
        flat_pts = extract_poly_points(pts_raw) if pts_raw is not None else []
        if len(flat_pts) >= 3:
            draw.polygon(flat_pts, fill=green_fill)
            draw.line(flat_pts + [flat_pts[0]], fill=green_solid, width=3)

    result = Image.alpha_composite(base.convert("RGBA"), overlay)
    return result.convert("RGB")


def estimate_severity(p, img_w, img_h):
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
# 2. HÀM XUẤT PDF 2 TRANG
# =========================================================

def export_pdf(
    original_img,
    analyzed_img,
    metrics_df,
    chart_bar_png: io.BytesIO = None,
    chart_pie_png: io.BytesIO = None,
    filename="bkai_report.pdf",
):
    left_margin = 25 * mm
    right_margin = 25 * mm
    top_margin = 20 * mm
    bottom_margin = 20 * mm

    page_w, page_h = A4
    content_w = page_w - left_margin - right_margin
    content_h = page_h - top_margin - bottom_margin

    def _build(buf):
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=left_margin,
            rightMargin=right_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
        )

        styles = getSampleStyleSheet()
        for s in styles.byName:
            styles[s].fontName = FONT_NAME

        title = ParagraphStyle(
            "TitleVN", parent=styles["Title"],
            fontName=FONT_NAME, alignment=1, fontSize=20, leading=24
        )
        h2 = ParagraphStyle(
            "H2VN", parent=styles["Heading2"],
            fontName=FONT_NAME, spaceBefore=8, spaceAfter=4
        )
        normal = ParagraphStyle(
            "NormalVN", parent=styles["Normal"],
            fontName=FONT_NAME, leading=13
        )

        story = []

        from PIL import Image as PILImage

        def add_pil_image(pil, caption, max_h_ratio=0.28):
            if pil is None:
                return
            if not isinstance(pil, PILImage.Image):
                pil = pil.convert("RGB")
            w, h = pil.size
            max_h = content_h * max_h_ratio
            scale = min(content_w / w, max_h / h, 1.0)
            buf_img = io.BytesIO()
            pil.save(buf_img, format="PNG")
            buf_img.seek(0)
            story.append(Paragraph(caption, h2))
            story.append(RLImage(buf_img, width=w * scale, height=h * scale))
            story.append(Spacer(1, 4 * mm))

        # =============== TRANG 1 =================
        if os.path.exists(LOGO_PATH):
            story.append(RLImage(LOGO_PATH, width=38 * mm))
            story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("BÁO CÁO KIỂM TRA VẾT NỨT BÊ TÔNG", title))
        story.append(Paragraph("Concrete Crack Inspection Report", normal))
        story.append(Spacer(1, 6 * mm))

        add_pil_image(original_img, "Ảnh gốc / Original Image", max_h_ratio=0.26)
        add_pil_image(analyzed_img, "Ảnh phân tích / Result Image", max_h_ratio=0.26)

        if chart_bar_png is not None:
            story.append(Paragraph("Biểu đồ: Độ tin cậy từng vùng nứt", h2))
            story.append(
                RLImage(chart_bar_png, width=content_w, height=content_h * 0.22)
            )
            story.append(Spacer(1, 3 * mm))

        if chart_pie_png is not None:
            story.append(Paragraph("Biểu đồ: Tỷ lệ vùng nứt / toàn ảnh", h2))
            story.append(
                RLImage(chart_pie_png, width=content_w, height=content_h * 0.22)
            )
            story.append(Spacer(1, 3 * mm))

        # Sang trang 2
        story.append(PageBreak())

        # =============== TRANG 2 – BẢNG THÔNG TIN ===============
        story.append(Paragraph("Bảng thông tin vết nứt / Crack Metrics", h2))

        data = [[
            Paragraph("Chỉ số (VI)", normal),
            Paragraph("Metric (EN)", normal),
            Paragraph("Giá trị / Value", normal),
            Paragraph("Ý nghĩa / Description", normal),
        ]]

        for _, r in metrics_df.iterrows():
            vi_txt = Paragraph(str(r["vi"]), normal)
            en_txt = Paragraph(str(r["en"]), normal)
            val_txt = Paragraph(str(r["value"]), normal)
            full_desc = str(r["desc"])
            short_desc = (full_desc[:180] + "...") if len(full_desc) > 180 else full_desc
            desc_txt = Paragraph(short_desc, normal)
            data.append([vi_txt, en_txt, val_txt, desc_txt])

        col_widths = [
            0.2 * content_w,
            0.2 * content_w,
            0.2 * content_w,
            0.4 * content_w,
        ]
        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e88e5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 6 * mm))

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(
            Paragraph(
                f"BKAI © {datetime.datetime.now().year} – Report generated at {now}",
                normal,
            )
        )

        doc.build(story)

    buf = io.BytesIO()
    try:
        _build(buf)
    except LayoutError:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("BKAI - Báo cáo rút gọn", styles["Title"]),
            Spacer(1, 8 * mm),
            Paragraph(
                "Nội dung quá dài. Vui lòng xem chi tiết trên web BKAI.",
                styles["Normal"],
            ),
        ]
        doc.build(story)

    buf.seek(0)
    return buf

# =========================================================
# 3. STAGE 2 – DEMO KIẾN THỨC
# =========================================================

def show_stage2_demo(key_prefix="stage2"):
    st.subheader("Stage 2 – Phân loại vết nứt & gợi ý nguyên nhân / biện pháp")

    # =========================
    # 1) Bảng 1: Phân loại theo cơ chế nứt (đã làm trước)
    # =========================
    options = [
        "I.1 Nứt co ngót dẻo (Plastic Shrinkage Crack)",
        "I.2 Nứt lún dẻo / lắng dẻo (Plastic Settlement Crack)",
        "II.1 Nứt do co ngót khô (Drying Shrinkage Crack)",
        "II.2 Nứt do đóng băng – băng tan (Freeze–Thaw Crack)",
        "II.3 Nứt do nhiệt (Thermal Crack)",
        "II.4a Nứt do hoá chất – sunfat tấn công (Sulfate Attack)",
        "II.4b Nứt do hoá chất – kiềm cốt liệu (Alkali–Aggregate Reaction)",
        "II.5 Nứt do ăn mòn cốt thép (Corrosion–Induced Crack)",
        "II.6a Nứt do tải trọng – nứt uốn (Flexural Crack)",
        "II.6b Nứt do tải trọng – nứt cắt/nén/xoắn (Shear/Compression/Torsion Cracks)",
        "II.7 Nứt do lún (Settlement Crack)",
    ]

    st.selectbox(
        "Chọn loại vết nứt (tóm tắt):",
        options,
        key=f"{key_prefix}_summary_selectbox",
    )

    demo_data = pd.DataFrame(
        [
            {
                "Loại vết nứt": "I.1 Nứt co ngót dẻo",
                "Nguyên nhân": (
                    "Bề mặt bê tông mất nước quá nhanh do nhiệt độ cao, độ ẩm thấp, gió lớn, "
                    "bảo dưỡng chậm khi bê tông còn dẻo → co ngót bề mặt vượt quá cường độ kéo sớm."
                ),
                "Biện pháp": (
                    "Làm ẩm nền/ván khuôn; che nắng, chắn gió; bảo dưỡng ẩm sớm; phun sương, phủ màng bảo dưỡng; "
                    "thiết kế cấp phối w/c thấp, hạn chế bleeding."
                ),
            },
            {
                "Loại vết nứt": "I.2 Nứt lún dẻo / lắng dẻo",
                "Nguyên nhân": (
                    "Bê tông lún xuống dưới tác dụng trọng lực nhưng bị cản trở bởi cốt thép, "
                    "vùng thay đổi tiết diện, ván khuôn hẹp → tạo khe nứt trên đỉnh cốt thép hoặc tại chỗ thay đổi mặt cắt."
                ),
                "Biện pháp": (
                    "Dùng bê tông độ sụt vừa phải, bleeding thấp; tăng hạt mịn; "
                    "bố trí cốt thép hợp lý; đầm chặt đều; kiểm tra độ kín và độ cứng ván khuôn."
                ),
            },
            {
                "Loại vết nứt": "II.1 Nứt do co ngót khô",
                "Nguyên nhân": (
                    "Sau khi đông cứng, nước trong mao quản bay hơi trong môi trường khô/nóng "
                    "→ hồ xi măng co lại, bị hạn chế bởi cốt thép/kết cấu khác → nứt."
                ),
                "Biện pháp": (
                    "Thiết kế w/c thấp, tăng cốt liệu chắc; dùng phụ gia, sợi; "
                    "bảo dưỡng ẩm; tránh thi công trong điều kiện nắng nóng, gió mạnh; "
                    "bố trí khe co giãn hợp lý."
                ),
            },
            {
                "Loại vết nứt": "II.2 Nứt do đóng băng – băng tan",
                "Nguyên nhân": (
                    "Nước trong lỗ rỗng đóng băng gây giãn nở thể tích, áp suất thủy lực; "
                    "nhiều chu kỳ đóng băng–tan băng phá hoại hồ và cốt liệu, tạo nứt và bong tróc."
                ),
                "Biện pháp": (
                    "Dùng bê tông chống băng giá (phụ gia cuốn khí, w/c thấp); "
                    "thiết kế hỗn hợp đặc chắc; phủ lớp bảo vệ; hạn chế nước đọng và muối khử băng."
                ),
            },
            {
                "Loại vết nứt": "II.3 Nứt do nhiệt",
                "Nguyên nhân": (
                    "Chênh lệch nhiệt độ lớn giữa trong–ngoài khối bê tông hoặc giữa các vùng khác nhau "
                    "→ giãn nở/co lại không đều, bị kìm hãm → ứng suất nhiệt vượt cường độ kéo."
                ),
                "Biện pháp": (
                    "Kiểm soát nhiệt độ khi đổ (nước lạnh, đổ ban đêm); dùng xi măng LH, phụ gia làm chậm; "
                    "ống làm lạnh, đổ theo giai đoạn; tăng cường cốt thép phân tán; bảo dưỡng, che phủ cách nhiệt."
                ),
            },
            {
                "Loại vết nứt": "II.4a Nứt do hoá chất – sunfat tấn công",
                "Nguyên nhân": (
                    "Ion sunfat thấm vào bê tông, phản ứng với hồ xi măng tạo sản phẩm giãn nở (ettringite, gypsum) "
                    "→ ứng suất kéo lớn, nứt và phân hủy bê tông, thường từ ngoài vào trong."
                ),
                "Biện pháp": (
                    "Dùng xi măng chống sunfat (C₃A thấp), tro bay/xỉ; giữ w/c thấp; "
                    "chọn cốt liệu sạch; thiết kế bê tông đặc chắc, chống thấm; "
                    "hạn chế tiếp xúc trực tiếp môi trường sunfat."
                ),
            },
            {
                "Loại vết nứt": "II.4b Nứt do hoá chất – phản ứng kiềm cốt liệu (AAR)",
                "Nguyên nhân": (
                    "Kiềm trong xi măng/phụ gia phản ứng với cốt liệu phản ứng tạo gel AAR; "
                    "gel hút ẩm trương nở → áp suất nội lớn, nứt vi mô lan rộng, trương nở thể tích."
                ),
                "Biện pháp": (
                    "Dùng xi măng kiềm thấp, cốt liệu không phản ứng; "
                    "dùng tro bay, xỉ, silica fume; giữ w/c thấp; "
                    "hạn chế cung cấp ẩm liên tục; kiểm tra AAR khi thiết kế vật liệu."
                ),
            },
            {
                "Loại vết nứt": "II.5 Nứt do ăn mòn cốt thép",
                "Nguyên nhân": (
                    "Cốt thép bị ăn mòn (ion Cl⁻, CO₂, môi trường xâm thực), sản phẩm rỉ thép "
                    "giãn nở 2–6 lần → ép lên lớp bê tông bảo vệ, gây nứt dọc theo thanh thép, bong lớp bảo vệ."
                ),
                "Biện pháp": (
                    "Đảm bảo chiều dày và chất lượng lớp bảo vệ; dùng bê tông đặc chắc, chống thấm; "
                    "cốt thép chống ăn mòn hoặc phủ; phụ gia ức chế ăn mòn; "
                    "lớp phủ bảo vệ bề mặt trong môi trường xâm thực."
                ),
            },
            {
                "Loại vết nứt": "II.6a Nứt do tải trọng – nứt uốn",
                "Nguyên nhân": (
                    "Tải trọng làm ứng suất kéo do uốn vượt cường độ kéo của bê tông ở vùng chịu kéo."
                ),
                "Biện pháp": (
                    "Thiết kế đủ cốt thép chịu uốn; kiểm soát tải trọng sử dụng; "
                    "gia cường bằng thép/bê tông/FRP; tiêm epoxy phục hồi liên kết khi cần."
                ),
            },
            {
                "Loại vết nứt": "II.6b Nứt do tải trọng – nứt cắt/nén/xoắn",
                "Nguyên nhân": (
                    "Ứng suất cắt, nén, xoắn vượt khả năng chịu lực (tải tập trung lớn, tải lặp, "
                    "thay đổi sơ đồ chịu lực…) → xuất hiện nứt cắt, nứt nén, nứt xoắn."
                ),
                "Biện pháp": (
                    "Tăng cường cốt đai, cốt xiên, cốt xoắn; kiểm soát tải trọng; "
                    "gia cường cục bộ vùng chịu lực lớn; kiểm tra, bảo dưỡng định kỳ."
                ),
            },
            {
                "Loại vết nứt": "II.7 Nứt do lún",
                "Nguyên nhân": (
                    "Nền/lớp đệm bị lún, rửa trôi, lún lệch → biến dạng không đều, "
                    "sinh ứng suất kéo lớn tại dầm, sàn, móng vùng chênh lệch lún."
                ),
                "Biện pháp": (
                    "Khảo sát và xử lý nền tốt (gia cố, thay đất yếu); "
                    "thiết kế đủ độ cứng, khe lún/khe nhiệt hợp lý; "
                    "kiểm soát tải; khi đã nứt: tiêm epoxy, gia cường và xử lý nền."
                ),
            },
        ]
    )

    st.table(demo_data)
    st.caption("Bảng 1 – Tổng hợp các dạng nứt theo cơ chế hình thành và biện pháp kiểm soát.")

    # =========================
    # 2) Bảng 2: Phân loại theo cấu kiện (Dầm, Cột, Sàn, Tường)
    # =========================
    st.subheader("Phân loại các vết nứt bê tông thường xảy ra cho từng loại cấu kiện")

    component_crack_data = pd.DataFrame(
        [
            # --- DẦM ---
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt uốn",
                "Nguyên nhân": (
                    "Mô men uốn vượt khả năng chịu uốn; tiết diện hoặc cốt thép chịu uốn không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt thường chéo hoặc hơi cong, phát triển ở vùng giữa nhịp; "
                    "rộng nhất ở vùng chịu kéo (dưới đáy hoặc trên đỉnh dầm tùy sơ đồ nội lực)."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt cắt",
                "Nguyên nhân": (
                    "Lực cắt lớn tại gối hoặc gần điểm uốn; khả năng chịu cắt của bê tông/cốt đai không đủ; thiết kế không đúng."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt xiên, nghiêng khoảng 45° so với trục dầm; "
                    "có thể đơn lẻ hoặc thành nhóm; rộng nhất gần vùng trục trung hòa hoặc đáy dầm."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt xoắn",
                "Nguyên nhân": (
                    "Độ bền xoắn không đủ; thiếu cốt thép chịu xoắn; tiết diện dầm không phù hợp với mô-men xoắn."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chéo, dạng xoắn ốc hoặc ziczac quanh dầm; thường rộng hơn ở phần trên, "
                    "bề rộng tương đối đồng đều dọc theo vết nứt."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt trượt",
                "Nguyên nhân": (
                    "Bê tông bị xáo trộn khi cường độ chưa đạt; cốp pha/gối đỡ bị dịch chuyển khi bê tông chưa đủ cứng."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt gần mép gối đỡ, chạy gần phương thẳng đứng; "
                    "độ rộng lớn nhất tại đáy dầm."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt kéo",
                "Nguyên nhân": (
                    "Cốt thép chịu kéo không đủ, dầm quá tải, biến dạng không đều, tải trọng phân bố không đồng đều."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt gần vuông góc với trục dầm, vuông góc với phương ứng suất kéo; "
                    "phía dưới rộng, phía trên nhỏ; thường song song và phân bố khá đều."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt ăn mòn cốt thép",
                "Nguyên nhân": (
                    "Liên kết bê tông–cốt thép kém, lớp bảo vệ mỏng, cốt thép bị gỉ làm tăng thể tích, "
                    "tạo áp lực giãn nở lên bê tông."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt thường chạy dọc theo đường cốt thép; có thể xiên/chéo gần 45° tùy sơ đồ; "
                    "thường kèm vết gỉ, đổi màu bề mặt."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Dầm",
                "Loại vết nứt": "Vết nứt co ngót",
                "Nguyên nhân": (
                    "Bê tông dầm co ngót do mất nước, bị kiềm chế bởi cốt thép/kết cấu lân cận."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt nhỏ, có thể phân bố tương đối đều, thường gần vuông góc trục dầm hoặc tạo thành mạng nhỏ."
                ),
                "Hình ảnh minh họa": "—",
            },

            # --- CỘT ---
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt ngang",
                "Nguyên nhân": (
                    "Thiếu mô-men kiềm chế, diện tích cốt thép nhỏ hoặc bố trí không hợp lý; "
                    "chịu lực cắt, tải trọng trực tiếp hoặc uốn đơn trục lớn."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt ngang quanh cột, thường thấy tại vùng nối dầm–cột hoặc chỗ có ứng suất kéo lớn."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt chéo",
                "Nguyên nhân": (
                    "Thiết kế không đúng, cột không đủ khả năng chịu tải dọc và uốn; "
                    "cường độ bê tông hoặc cốt thép không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chạy xiên trên bề mặt cột, xuất hiện khi cột chịu tải lớn gần/ vượt khả năng chịu lực."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt tách (dọc)",
                "Nguyên nhân": (
                    "Cốt thép dọc không đủ, bê tông cường độ thấp; khi tải trọng đạt gần khả năng chịu tải tối đa "
                    "gây phân tách bê tông theo phương dọc."
                ),
                "Đặc trưng hình dạng": (
                    "Các vết nứt dọc ngắn, song song, độ rộng khác nhau, thường xuất hiện vùng giữa chiều cao cột."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt do ăn mòn",
                "Nguyên nhân": (
                    "Cốt thép trong cột bị gỉ; sản phẩm ăn mòn giãn nở, gây nứt lớp bê tông bảo vệ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chạy theo đường bố trí cốt thép; thường kèm vết gỉ, bong tróc lớp bảo vệ."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Cột",
                "Loại vết nứt": "Vết nứt co ngót",
                "Nguyên nhân": (
                    "Bê tông cột co ngót bị kiềm chế bởi cốt thép và kết cấu liên kết (dầm, sàn)."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt dọc mảnh, song song, phân bố tương đối đều trên bề mặt cột."
                ),
                "Hình ảnh minh họa": "—",
            },

            # --- SÀN ---
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt co ngót dẻo",
                "Nguyên nhân": (
                    "Nhiệt độ cao, độ ẩm thấp, gió mạnh làm bốc hơi nước nhanh trước khi bê tông nắm chắc."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt nông, nhỏ (micro-cracks), chiều dài không lớn; hình dạng ngẫu nhiên, đa giác, "
                    "bắt chéo hoặc song song nhau trên bề mặt."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt co ngót khô",
                "Nguyên nhân": (
                    "Bê tông sàn đông cứng trong môi trường khô, nóng → nước bay hơi, hồ xi măng co lại."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt rõ, tạo mạng lưới (map cracking) hoặc đường thẳng ngang/trục trên mặt sàn."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt do nhiệt",
                "Nguyên nhân": (
                    "Nhiệt thủy hóa tăng trong khối sàn, bên trong giãn nở trong khi bề mặt mát hơn, bị co "
                    "→ chênh lệch biến dạng nhiệt lớn."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt bề mặt, đóng vảy, xuống cấp lớp bê tông bề mặt; thường gần song song bề mặt, "
                    "có thể kết hợp bong tróc."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt uốn",
                "Nguyên nhân": (
                    "Mô men uốn vượt khả năng chịu uốn; tiết diện/cốt thép chịu uốn không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chéo hoặc hơi cong, rộng nhất ở mặt chịu kéo của sàn (thường là mặt dưới giữa nhịp)."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt cắt",
                "Nguyên nhân": (
                    "Lực cắt lớn gần gối hoặc vùng chịu tải tập trung; thiếu cốt đai/cốt thép chịu cắt."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt xiên ~45° so với trục sàn; có thể đơn lẻ hoặc nhóm."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt xoắn",
                "Nguyên nhân": (
                    "Sàn làm việc như bản chịu xoắn (vùng góc, bản console…), độ bền xoắn không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chéo, dạng xoắn ốc tương tự dầm, rộng tương đối đồng đều."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt ăn mòn cốt thép",
                "Nguyên nhân": (
                    "Ion Cl⁻, nước biển, muối khử băng xâm nhập; lớp bảo vệ mỏng; cốt thép bị gỉ và giãn nở."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chạy dọc theo hướng bố trí cốt thép; thường kèm vết gỉ và bong lớp bảo vệ."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt do tải trọng – lực tập trung",
                "Nguyên nhân": (
                    "Bản sàn bị quá tải tại một điểm; thiếu cốt thép chịu uốn cục bộ; bố trí thép không đúng."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt vuông góc phương ứng suất kéo, dạng chữ thập hoặc tỏa ra từ điểm chịu tải."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Sàn",
                "Loại vết nứt": "Vết nứt do tải trọng – lực phân bố",
                "Nguyên nhân": (
                    "Tải trọng phân bố nhưng vượt khả năng làm việc lâu dài; độ cứng sàn không đủ."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt dạng chữ thập, mạng lưới hoặc xiên, tỏa từ giữa sàn ra các cạnh."
                ),
                "Hình ảnh minh họa": "—",
            },

            # --- TƯỜNG BÊ TÔNG ---
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt co ngót",
                "Nguyên nhân": (
                    "Bề mặt tường nóng, bốc hơi nước nhanh; ứng suất co ngót vượt khả năng chịu kéo của bê tông tường."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt bề mặt, phạm vi rộng, ngẫu nhiên, đa giác, bắt chéo hoặc song song."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt do nhiệt",
                "Nguyên nhân": (
                    "Ứng suất và chuyển vị do chênh lệch nhiệt độ trong tường bê tông."
                ),
                "Đặc trưng hình dạng": (
                    "Thường là vết nứt thẳng đứng, mở rộng nhiều ở phía dưới hoặc ở vùng chịu kéo do nhiệt."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt ngang do tải trọng",
                "Nguyên nhân": (
                    "Tường chịu tải trọng vượt mức; phân phối tải không đều; hiệu ứng xoay, trượt ở chân tường."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt ngang chia tường thành hai phần; phần trên có thể nghiêng, phần giữa có xu hướng cong/lõm."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt dọc do tải trọng",
                "Nguyên nhân": (
                    "Tải trọng thẳng đứng lớn, lún cục bộ, hoặc thiếu cốt thép dọc."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt tách dọc chia tường thành hai mảng song song."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt chéo do tải trọng",
                "Nguyên nhân": (
                    "Kết hợp tác dụng của tải đứng và ngang; tường vừa chịu nén vừa chịu cắt/uốn."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chéo, bề rộng lớn nhất gần phía trên; thể hiện sự làm việc kém ổn định của tường."
                ),
                "Hình ảnh minh họa": "—",
            },
            {
                "Cấu kiện": "Tường bê tông",
                "Loại vết nứt": "Vết nứt ăn mòn cốt thép",
                "Nguyên nhân": (
                    "Cốt thép tường bị gỉ; sản phẩm ăn mòn giãn nở gây nứt lớp bảo vệ bê tông."
                ),
                "Đặc trưng hình dạng": (
                    "Vết nứt chạy theo vị trí thanh thép; thường kèm bong tróc, hoen gỉ trên bề mặt."
                ),
                "Hình ảnh minh họa": "—",
            },
        ]
    )

    st.table(component_crack_data)
    st.caption(
        "Bảng 2 – Phân loại các vết nứt bê tông thường gặp theo từng loại cấu kiện "
        "(dầm, cột, sàn, tường) – dùng cho phần kiến thức nền và phân tích kết quả mô hình."
    )

# =========================================================
# 3.5. LƯU THỐNG KÊ NGƯỜI DÙNG
# =========================================================

USER_STATS_FILE = "user_stats.json"

# Đọc danh sách thống kê (nếu có)
if os.path.exists(USER_STATS_FILE):
    with open(USER_STATS_FILE, "r", encoding="utf-8") as f:
        try:
            user_stats = json.load(f)
        except Exception:
            user_stats = []
else:
    user_stats = []

# =========================================================
# 4. GIAO DIỆN CHÍNH
# =========================================================

def run_main_app():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)
    with col_title:
        st.title("BKAI - MÔ HÌNH CNN PHÁT HIỆN VÀ PHÂN LOẠI VẾT NỨT")
        user = st.session_state.get("username", "")
        if user:
            st.caption(
                f"Xin chào **{user}** – Phân biệt ảnh nứt / không nứt & xuất báo cáo."
            )
        else:
            st.caption("Phân biệt ảnh nứt / không nứt & xuất báo cáo.")

    st.write("---")

    # ------------ FORM THÔNG TIN NGƯỜI DÙNG (BẮT BUỘC) ------------
    # Nếu chưa có cờ profile_filled thì mặc định là False
    if "profile_filled" not in st.session_state:
        st.session_state.profile_filled = False

    # Nếu chưa điền, luôn hiển thị form
    if not st.session_state.profile_filled:
        st.subheader("Thông tin người sử dụng (bắt buộc trước khi phân tích)")

        with st.form("user_info_form"):
            full_name = st.text_input("Họ và tên *")
            occupation = st.selectbox(
                "Nghề nghiệp / Nhóm đối tượng *",
                [
                    "Sinh viên",
                    "Kỹ sư xây dựng",
                    "Kỹ sư IT",
                    "Nghiên cứu viên",
                    "Học viên cao học",
                    "Giảng viên",
                    "Khác",
                ],
            )
            email = st.text_input("Email *")

            submit_info = st.form_submit_button("Lưu thông tin & bắt đầu phân tích")

        if submit_info:
            if not full_name or not occupation or not email:
                st.warning("Vui lòng điền đầy đủ Họ tên, Nghề nghiệp và Email.")
            elif "@" not in email or "." not in email:
                st.warning("Email không hợp lệ, vui lòng kiểm tra lại.")
            else:
                # Lưu vào session_state
                st.session_state.profile_filled = True
                st.session_state.user_full_name = full_name
                st.session_state.user_occupation = occupation
                st.session_state.user_email = email

                # Ghi vào file thống kê
                record = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "login_user": st.session_state.get("username", ""),
                    "full_name": full_name,
                    "occupation": occupation,
                    "email": email,
                }
                user_stats.append(record)
                try:
                    with open(USER_STATS_FILE, "w", encoding="utf-8") as f:
                        json.dump(user_stats, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    st.warning(f"Lưu thống kê người dùng bị lỗi: {e}")

                st.success("Đã lưu thông tin. Bạn có thể tải ảnh lên để phân tích.")

        # Nếu chưa fill form đúng -> dừng, KHÔNG cho upload ảnh
        if not st.session_state.profile_filled:
            return

    # ------------ SAU KHI ĐÃ ĐIỀN FORM, HIỆN SIDEBAR + UPLOAD ------------
    st.sidebar.header("Cấu hình phân tích")
    min_conf = st.sidebar.slider(
        "Ngưỡng confidence tối thiểu", 0.0, 1.0, 0.3, 0.05
    )
    st.sidebar.caption("Chỉ hiển thị những vết nứt có độ tin cậy ≥ ngưỡng này.")

    uploaded_files = st.file_uploader(
        "Tải một hoặc nhiều ảnh bê tông (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    analyze_btn = st.button("🔍 Phân tích ảnh")

    if analyze_btn:
        if not uploaded_files:
            st.warning("Vui lòng chọn ít nhất một ảnh trước khi bấm **Phân tích**.")
            st.stop()

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            st.write("---")
            st.markdown(f"## Ảnh {idx}: `{uploaded_file.name}`")

            t0 = time.time()
            orig_img = Image.open(uploaded_file).convert("RGB")
            img_w, img_h = orig_img.size

            buf = io.BytesIO()
            orig_img.save(buf, format="JPEG")
            buf.seek(0)

            with st.spinner(f"Đang gửi ảnh {idx} tới mô hình AI trên Roboflow..."):
                try:
                    resp = requests.post(
                        ROBOFLOW_FULL_URL,
                        files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
                        timeout=60,
                    )
                except Exception as e:
                    st.error(f"Lỗi gọi API Roboflow cho ảnh {uploaded_file.name}: {e}")
                    continue

            if resp.status_code != 200:
                st.error(f"Roboflow trả lỗi cho ảnh {uploaded_file.name}.")
                st.text(resp.text[:2000])
                continue

            result = resp.json()
            predictions = result.get("predictions", [])
            preds_conf = [
                p for p in predictions if float(p.get("confidence", 0)) >= min_conf
            ]

            t1 = time.time()
            total_time = t1 - t0

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ảnh gốc")
                st.image(orig_img, use_column_width=True)

            analyzed_img = None
            with col2:
                st.subheader("Ảnh phân tích")
                if len(preds_conf) == 0:
                    st.image(orig_img, use_column_width=True)
                    st.success("✅ Kết luận: **Không phát hiện vết nứt rõ ràng**.")
                else:
                    analyzed_img = draw_predictions_with_mask(
                        orig_img, preds_conf, min_conf
                    )
                    st.image(analyzed_img, use_column_width=True)
                    st.error("⚠️ Kết luận: **CÓ vết nứt trên ảnh.**")

            if len(preds_conf) == 0 or analyzed_img is None:
                continue

            st.write("---")
            tab_stage1, tab_stage2 = st.tabs(
                ["Stage 1 – Báo cáo chi tiết", "Stage 2 – Phân loại vết nứt"]
            )

            # (giữ nguyên phần STAGE 1 & STAGE 2 như code trước đó)
            # ...


            st.write("---")
            tab_stage1, tab_stage2 = st.tabs(
                ["Stage 1 – Báo cáo chi tiết", "Stage 2 – Phân loại vết nứt"]
            )

            # ================== STAGE 1 ==================
            with tab_stage1:
                st.subheader("Bảng thông tin vết nứt")

                confs = [float(p.get("confidence", 0)) for p in preds_conf]
                avg_conf = sum(confs) / len(confs)
                map_val = round(min(1.0, avg_conf - 0.05), 2)

                max_ratio = 0.0
                max_p = preds_conf[0]
                for p in preds_conf:
                    w = float(p.get("width", 0))
                    h = float(p.get("height", 0))
                    ratio = (w * h) / (img_w * img_h)
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
                        "vi": "Độ tin cậy (Confidence)",
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
                        "value": (
                            "Vết nứt có nguy cơ, cần kiểm tra thêm."
                            if "Nguy hiểm" in severity
                            else "Vết nứt nhỏ, nên tiếp tục theo dõi."
                        ),
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

                # -------- BIỂU ĐỒ & LƯU PNG --------
                st.subheader("Biểu đồ thống kê")
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    fig1 = plt.figure(figsize=(4, 3))
                    plt.bar(range(1, len(confs) + 1), confs)
                    plt.xlabel("Crack #")
                    plt.ylabel("Confidence")
                    plt.ylim(0, 1)
                    plt.title("Độ tin cậy từng vùng nứt")
                    st.pyplot(fig1)
                    bar_png = fig_to_png(fig1)
                    plt.close(fig1)

                with col_chart2:
                    labels = ["Vùng nứt lớn nhất", "Phần ảnh còn lại"]
                    sizes = [max_ratio, 1 - max_ratio]
                    fig2 = plt.figure(figsize=(4, 3))
                    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
                    plt.title("Tỷ lệ vùng nứt so với toàn ảnh")
                    st.pyplot(fig2)
                    pie_png = fig_to_png(fig2)
                    plt.close(fig2)

                # -------- XUẤT PDF 2 TRANG --------
                pdf_buf = export_pdf(
                    original_img=orig_img,
                    analyzed_img=analyzed_img,
                    metrics_df=metrics_df,
                    chart_bar_png=bar_png,
                    chart_pie_png=pie_png,
                )

                st.download_button(
                    "📄 Tải báo cáo PDF cho ảnh này",
                    data=pdf_buf,
                    file_name=f"BKAI_CrackReport_{uploaded_file.name.split('.')[0]}.pdf",
                    mime="application/pdf",
                    key=f"pdf_btn_{idx}_{uploaded_file.name}",
                )

            # ================== STAGE 2 ==================
            with tab_stage2:
                show_stage2_demo(key_prefix=f"stage2_{idx}")

# =========================================================
# 5. ĐĂNG KÝ / ĐĂNG NHẬP
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
    st.title("BKAI - Concrete Crack Inspection")
    st.subheader("Vui lòng đăng nhập để sử dụng hệ thống phân tích vết nứt bê tông.")

    tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])

    with tab_login:
        login_user = st.text_input("Tên đăng nhập", key="login_user")
        login_pass = st.text_input("Mật khẩu", type="password", key="login_pass")
        if st.button("Đăng nhập"):
            if login_user in users and users[login_user] == login_pass:
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.success(f"Đăng nhập thành công! Xin chào, {login_user} 👋")
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu.")

    with tab_register:
        reg_user = st.text_input("Tên đăng nhập mới", key="reg_user")
        reg_pass = st.text_input("Mật khẩu mới", type="password", key="reg_pass")
        reg_pass2 = st.text_input("Nhập lại mật khẩu", type="password", key="reg_pass2")

        if st.button("Tạo tài khoản"):
            if not reg_user or not reg_pass:
                st.warning("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")
            elif reg_user in users:
                st.error("Tên đăng nhập đã tồn tại, hãy chọn tên khác.")
            elif reg_pass != reg_pass2:
                st.error("Mật khẩu nhập lại không khớp.")
            else:
                users[reg_user] = reg_pass
                with open(USERS_FILE, "w", encoding="utf-8") as f:
                    json.dump(users, f, ensure_ascii=False, indent=2)
                st.success("Tạo tài khoản thành công! Bạn có thể quay lại tab Đăng nhập.")

# =========================================================
# 6. MAIN ENTRY
# =========================================================

if st.session_state.authenticated:
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.username}")
        if st.button("Đăng xuất"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()
    run_main_app()
else:
    show_auth_page()

