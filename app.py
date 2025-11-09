import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import time
import datetime
import os
import pandas as pd
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
from reportlab.platypus.doctemplate import LayoutError

# =========================================================
# 0. CẤU HÌNH CHUNG
# =========================================================

# --- 0.1. Roboflow URL (BẮT BUỘC SỬA CHO ĐÚNG MODEL CỦA BẠN) ---
ROBOFLOW_FULL_URL = (
    "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"
)

# --- 0.2. Logo BKAI (ảnh PNG đặt cạnh file app.py) ---
LOGO_PATH = "BKAI_Logo.png"

# --- 0.3. Font Unicode cho PDF ---
FONT_PATH = "times.ttf"          # nếu bạn có file Times New Roman -> đặt tên này
FONT_NAME = "TimesVN"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
else:
    # Fallback sang DejaVuSans có sẵn trên server
    FONT_NAME = "DejaVu"
    pdfmetrics.registerFont(
        TTFont(FONT_NAME, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )

# =========================================================
# 1. CÁC HÀM XỬ LÝ ẢNH
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
    Vẽ ảnh phân tích với:
      - Box
      - Label
      - Vùng mask (polygon) 

    TẤT CẢ dùng cùng 1 màu xanh lá.
    """
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Màu xanh lá
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

        # Tính box
        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        # Box xanh lá
        draw.rectangle([x0, y0, x1, y1], outline=green_solid, width=3)

        # Nhãn trên mép box
        cls = p.get("class", "crack")
        label = f"{cls} {conf:.2f}"
        text_pos = (x0 + 3, y0 + 3)
        draw.text(text_pos, label, fill=green_solid)

        # Polyline + mask cùng màu xanh
        pts_raw = p.get("points")
        flat_pts = extract_poly_points(pts_raw) if pts_raw is not None else []
        if len(flat_pts) >= 3:
            # Vùng mask trong suốt màu xanh
            draw.polygon(flat_pts, fill=green_fill)
            # Outline polygon màu xanh
            draw.line(flat_pts + [flat_pts[0]], fill=green_solid, width=3)

    # Ghép overlay lên ảnh gốc
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
# 2. HÀM XUẤT PDF
# =========================================================


def export_pdf(original_img, analyzed_img, metrics_df, filename="bkai_report.pdf"):
    """Tạo file PDF báo cáo, dùng font Unicode (TimesVN/DejaVu),
    đã giới hạn kích thước ảnh + bọc bảng bằng Paragraph để tránh LayoutError.
    Nếu vẫn lỗi thì sinh một PDF rút gọn.
    """

    # ---------- CẤU HÌNH DOC & KHUNG NỘI DUNG ----------
    left_margin = 25 * mm
    right_margin = 25 * mm
    top_margin = 20 * mm
    bottom_margin = 20 * mm

    page_w, page_h = A4
    content_width = page_w - left_margin - right_margin
    content_height = page_h - top_margin - bottom_margin

    # Hàm xây PDF chính, để có thể gọi lại khi cần
    def build_story(buf):
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
        normal = ParagraphStyle(
            "NormalVN",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            leading=13,
        )

        story = []

        # ===== Hàm phụ: thêm ảnh và tự scale cho vừa khung (chỉ chiếm tối đa 40% chiều cao) =====
        from PIL import Image as PILImage

        def add_pil_image(pil_img, title_text):
            if pil_img is None:
                return

            if not isinstance(pil_img, PILImage.Image):
                pil_img = pil_img.convert("RGB")

            w, h = pil_img.size
            # Giới hạn rất an toàn: rộng <= content_width, cao <= 0.4 * content_height
            max_h = content_height * 0.4
            scale = min(content_width / w, max_h / h, 1.0)

            img_buf = io.BytesIO()
            pil_img.save(img_buf, format="PNG")
            img_buf.seek(0)

            story.append(Paragraph(title_text, h2))
            story.append(Spacer(1, 4 * mm))
            story.append(
                RLImage(
                    img_buf,
                    width=w * scale,
                    height=h * scale,
                )
            )
            story.append(Spacer(1, 6 * mm))

        # ===== Logo + tiêu đề =====
        if os.path.exists(LOGO_PATH):
            story.append(RLImage(LOGO_PATH, width=40 * mm))
            story.append(Spacer(1, 6 * mm))

        story.append(Paragraph("BÁO CÁO KIỂM TRA VẾT NỨT BÊ TÔNG", title_style))
        story.append(Paragraph("Concrete Crack Inspection Report", normal))
        story.append(Spacer(1, 8 * mm))

        # ===== Ảnh gốc =====
        add_pil_image(original_img, "Ảnh gốc / Original Image")

        # ===== Ảnh kết quả =====
        add_pil_image(analyzed_img, "Ảnh phân tích / Result Image")

        # ===== Bảng metrics =====
        story.append(Paragraph("Bảng thông tin vết nứt / Crack Metrics", h2))

        # Header
        data = [[
            Paragraph("Chỉ số (VI)", normal),
            Paragraph("Metric (EN)", normal),
            Paragraph("Giá trị / Value", normal),
            Paragraph("Ý nghĩa / Description", normal),
        ]]

        # Các dòng dữ liệu: dùng Paragraph để tự wrap
        for _, row in metrics_df.iterrows():
            vi_txt = Paragraph(str(row["vi"]), normal)
            en_txt = Paragraph(str(row["en"]), normal)
            val_txt = Paragraph(str(row["value"]), normal)
            desc_txt = Paragraph(str(row["desc"]), normal)
            data.append([vi_txt, en_txt, val_txt, desc_txt])

        # Chia content_width cho 4 cột (0.2, 0.2, 0.2, 0.4)
        col_widths = [
            0.2 * content_width,
            0.2 * content_width,
            0.2 * content_width,
            0.4 * content_width,
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
        story.append(Spacer(1, 8 * mm))

        # ===== Footer =====
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(
            Paragraph(
                f"BKAI © {datetime.datetime.now().year} – Report generated at {now_str}",
                normal,
            )
        )

        doc.build(story)

    # --------- THỬ BUILD BẢN ĐẦY ĐỦ, NẾU LỖI THÌ LÀM BẢN RÚT GỌN ----------
    buf = io.BytesIO()
    try:
        build_story(buf)
    except LayoutError:
        # Nếu vẫn LayoutError (ảnh/bảng quá dị), sinh file PDF tối giản
        buf = io.BytesIO()
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
        normal = styles["Normal"]
        title_style = styles["Title"]

        story = []
        story.append(Paragraph("BKAI - Báo cáo rút gọn", title_style))
        story.append(Spacer(1, 10 * mm))
        story.append(
            Paragraph(
                "Nội dung chi tiết (ảnh hoặc bảng) quá lớn so với khổ giấy nên không thể hiển thị đầy đủ trong PDF. "
                "Vui lòng xem chi tiết trực tiếp trên giao diện web BKAI.",
                normal,
            )
        )
        doc.build(story)

    buf.seek(0)
    return buf



# =========================================================
# 3. HÀM STAGE 2 (DEMO)
# =========================================================


def show_stage2_demo(key_prefix="stage2"):
    """Stage 2 demo: phân loại vết nứt & gợi ý nguyên nhân / biện pháp."""
    st.subheader("Stage 2– Phân loại vết nứt & gợi ý nguyên nhân / biện pháp")

    options = [
        "Vết nứt dọc (Longitudinal Crack)",
        "Vết nứt ngang (Transverse Crack)",
        "Vết nứt mạng (Map Crack)",
    ]

    # THÊM key để tránh trùng ID
    selected_label = st.selectbox(
        "Chọn loại vết nứt:",
        options,
        key=f"{key_prefix}_selectbox",
    )

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
    st.caption("Stage 2 hiện tại chỉ là demo – bảng kiến thức cơ bản về các dạng vết nứt.")


    # =========================================================
    # 2. BẢNG KIẾN THỨC CHI TIẾT TỪ LUẬN VĂN CỦA BẠN
    # =========================================================
    st.markdown("### 📚 Bảng kiến thức chi tiết về các dạng vết nứt bê tông")

    detailed_cracks = [
        # I. TRẠNG THÁI BÊ TÔNG TRƯỚC KHI ĐÔNG CỨNG
        {
            "Nhóm": "I. Trước khi đông cứng",
            "Loại vết nứt": "Nứt co ngót dẻo",
            "Nguyên nhân hình thành": """
Nứt co ngót dẻo xảy ra khi nhiệt độ không khí cao, độ ẩm tương đối thấp và vận tốc gió lớn làm tốc độ bay hơi nước trên bề mặt bê tông vượt quá tốc độ nước dâng lên. 
Mất cân bằng ẩm tạo ứng suất kéo; nếu vượt cường độ kéo sớm của bê tông, vết nứt hình thành ngay trong giai đoạn đầu bảo dưỡng.""",
            "Đặc trưng hình dạng / hình học": """
Vết nứt bề mặt, phạm vi rộng, hình dạng ngẫu nhiên: đa giác, bắt chéo nhau hoặc song song. 
Ban đầu nứt nhỏ, sau có thể phát triển sâu toàn bộ chiều dày bản/ sàn.""",
            "Thời gian xuất hiện": "Từ ~30 phút đến 6 giờ sau khi đổ bê tông.",
            "Cách kiểm soát / phòng ngừa": """
- Làm ẩm nền, ván khuôn, bề mặt trước và sau khi đổ bê tông. 
- Dựng mái che nắng, rào chắn gió, rút ngắn thời gian từ đổ đến bảo dưỡng.
- Phủ bề mặt bằng tấm nhựa, vải ẩm, phun sương mù bão hòa không khí trên bề mặt.
- Áp dụng chế độ bảo dưỡng ẩm liên tục, tránh dùng quá nhiều khói silic làm tăng tốc độ mất nước."""
        },
        {
            "Nhóm": "I. Trước khi đông cứng",
            "Loại vết nứt": "Nứt do lún dẻo (lắng dẻo)",
            "Nguyên nhân hình thành": """
Bê tông tươi có xu hướng lún dưới tác dụng trọng lực trong quá trình đóng rắn. 
Nếu quá trình lắng bị cản trở bởi cốt thép đặt gần bề mặt, thay đổi tiết diện, ván khuôn hẹp hoặc không chắc chắn, sẽ tạo nên chênh lệch chuyển vị và hình thành vết nứt lún dẻo.""",
            "Đặc trưng hình dạng / hình học": """
Vết nứt có độ rộng lớn hơn ở bề mặt và thu hẹp dần về phía cốt thép hoặc chỗ thay đổi tiết diện. 
Thường xuất hiện dọc theo thanh cốt thép gần đỉnh, dưới chân cột loe, vùng giao tiếp dầm – cột...""",
            "Thời gian xuất hiện": "Từ khoảng 10 phút đến 3 giờ sau khi đổ.",
            "Cách kiểm soát / phòng ngừa": """
- Dùng hỗn hợp bê tông độ sụt vừa phải, kết dính tốt, hàm lượng hạt mịn đủ. 
- Kiểm soát tỷ lệ nước, đầm chặt kỹ để giảm lún không đều.
- Bố trí cốt thép hợp lý, tránh đặt quá sát bề mặt hoặc chỗ thay đổi tiết diện đột ngột.
- Thiết kế hợp lý vùng dầm – cột, gối – dầm, điều chỉnh phương án thi công để giảm chênh lệch lún."""
        },

        # II. TRẠNG THÁI SAU KHI ĐÔNG CỨNG – CƠ NGUYÊN
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do co ngót khô",
            "Nguyên nhân hình thành": """
Co ngót khô là hiện tượng giảm thể tích do nước trong lỗ rỗng và mao quản bay hơi khi bê tông tiếp xúc môi trường khô, nóng. 
Xi măng và hồ vữa co lại, trong khi cốt liệu hạn chế biến dạng, tạo ứng suất kéo nội bộ. 
Nếu w/c cao, bê tông xốp, co ngót càng lớn.""",
            "Đặc trưng hình dạng / hình học": """
Vết nứt thường lớn, sâu, kéo dài theo phương ngang/dọc hoặc dạng mạng lưới trên bề mặt. 
Độ sâu có thể từ vài mm đến vài cm tùy mức độ co ngót và chiều dày cấu kiện.""",
            "Thời gian xuất hiện": "Sau vài tuần đến vài tháng.",
            "Cách kiểm soát / phòng ngừa": """
- Thiết kế bê tông với tỷ lệ w/c thấp, tăng cốt liệu lớn, chắc để hạn chế co ngót. 
- Dùng phụ gia khoáng, sợi (polyme, PP...) phân tán ứng suất.
- Bố trí khe co giãn với khoảng cách hợp lý.
- Bảo dưỡng ẩm đúng cách, tránh thi công trong điều kiện quá khô, nóng, gió lớn."""
        },
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do đóng băng – băng tan",
            "Nguyên nhân hình thành": """
Nước trong lỗ rỗng khi đóng băng giãn nở thể tích, tạo áp suất thủy lực lên hồ xi măng. 
Lặp lại nhiều chu kỳ đóng băng – tan băng gây phá hoại dần, vượt quá cường độ kéo của bê tông và tạo vết nứt dưới bề mặt, nứt bong bật bề mặt.""",
            "Đặc trưng hình dạng / hình học": """
Biểu hiện dưới dạng nứt, đóng vảy, bong bật từng mảng nhỏ trên bề mặt (spalling). 
Vết bật ra thường có dạng tròn, đường kính vài mm đến ~100 mm, sâu tới 40 mm, bề mặt bê tông xuống cấp chung.""",
            "Thời gian xuất hiện": "Sau 1 hoặc nhiều mùa đông (chu kỳ đóng băng – tan băng).",
            "Cách kiểm soát / phòng ngừa": """
- Dùng bê tông chống băng giá: w/c thấp, cuốn khí để tạo vi bọt giảm áp suất trong lỗ rỗng.
- Bảo vệ bề mặt bằng lớp chống thấm, hạn chế nước thấm sâu.
- Hạn chế, kiểm soát sử dụng muối khử băng (NaCl, CaCl₂...) trên bề mặt bê tông."""
        },
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do nhiệt (biến dạng nhiệt độ)",
            "Nguyên nhân hình thành": """
Khi bê tông chịu biến động nhiệt độ (mặt trời, khí hậu, thủy hóa trong khối lớn), các phần khác nhau giãn nở/co lại khác nhau. 
Chênh lệch nhiệt độ lớn giữa lõi và bề mặt hoặc giữa các vùng của kết cấu tạo ứng suất nhiệt; nếu vượt cường độ kéo sẽ gây nứt nhiệt.""",
            "Đặc trưng hình dạng / hình học": """
Thường dưới dạng các vết nứt song song với bề mặt, có thể dạng đóng vảy, xuống cấp lớp bê tông bề mặt. 
Ở khối lớn: vết nứt có thể chạy theo phương ngang/dọc tương đối thẳng, kích thước lớn.""",
            "Thời gian xuất hiện": "Từ 1 ngày đến vài tuần sau khi đổ, đặc biệt ở khối bê tông lớn.",
            "Cách kiểm soát / phòng ngừa": """
- Dùng xi măng tỏa nhiệt thấp, phụ gia làm chậm, chia nhỏ khối đổ, dùng ống làm mát trong khối lớn.
- Kiểm soát nhiệt độ bê tông tươi (nước lạnh, che nắng, cách nhiệt).
- Tăng cường cốt thép phân bố để chịu ứng suất nhiệt.
- Bảo dưỡng đầy đủ, tránh để bề mặt nguội quá nhanh so với lõi khối bê tông."""
        },
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do hóa chất – sunfat tấn công",
            "Nguyên nhân hình thành": """
Ion sunfat (Na₂SO₄, K₂SO₄, MgSO₄, CaSO₄...) trong nước/đất thấm vào bê tông và phản ứng với sản phẩm thủy hóa xi măng. 
Sản phẩm giãn nở (ettringite, gypsum) tạo ứng suất lớn trong hồ xi măng, vượt quá cường độ kéo và gây nứt, phân rã bê tông. 
Bê tông xốp (w/c cao) và môi trường sunfat mạnh làm tăng nguy cơ hư hỏng.""",
            "Đặc trưng hình dạng / hình học": """
Vết nứt thường bắt đầu từ vùng tiếp xúc với môi trường sunfat (chân cột, móng, kết cấu ngập nước) rồi lan dần vào trong. 
Bề mặt bong rộp, mềm yếu, có thể kèm phồng rộp, vỡ cạnh, tách lớp.""",
            "Thời gian xuất hiện": "Từ 1 đến 5 năm (hoặc lâu hơn, tùy nồng độ sunfat và chất lượng bê tông).",
            "Cách kiểm soát / phòng ngừa": """
- Dùng xi măng chống sunfat (C₃A < 5%), kết hợp phụ gia khoáng (tro bay, xỉ lò cao) để giảm tính thấm. 
- Giữ w/c thấp (≤ 0,40), dùng phụ gia giảm nước.
- Thiết kế bê tông đặc chắc, chọn cốt liệu sạch, không chứa sunfat.
- Hạn chế bê tông tiếp xúc trực tiếp nước/môi trường giàu sunfat hoặc có lớp bảo vệ, chống thấm."""
        },
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do hóa chất – phản ứng kiềm–cốt liệu (AAR)",
            "Nguyên nhân hình thành": """
AAR là phản ứng giữa kiềm (Na₂Oeq) trong xi măng và khoáng phản ứng trong cốt liệu, tạo gel kiềm–silic. 
Khi có ẩm, gel trương nở, gây biến dạng kéo dài vượt khả năng chịu kéo của bê tông, tạo nên hệ thống vết nứt nội bộ, lan ra bề mặt.""",
            "Đặc trưng hình dạng / hình học": """
Các vết nứt thường dạng mạng lưới, chiều rộng từ vài mm đến vài cm, phát triển từ bên trong ra bề mặt. 
Có thể kèm hiện tượng trương nở, phồng, cong vênh kết cấu.""",
            "Thời gian xuất hiện": "Thường > 5 năm; có thể nhanh hơn (vài tuần–vài tháng) nếu vật liệu rất phản ứng.",
            "Cách kiểm soát / phòng ngừa": """
- Chọn cốt liệu không phản ứng hoặc đã kiểm soát AAR.
- Hạn chế kiềm của hệ (xi măng kiềm thấp, kết hợp phụ gia khoáng). 
- Giảm thấm nước để hạn chế ẩm nuôi gel.
- Thiết kế, thi công và bảo dưỡng đúng quy trình, tránh nhiệt độ quá cao giai đoạn đầu."""
        },
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do ăn mòn cốt thép",
            "Nguyên nhân hình thành": """
Khi cốt thép bị ăn mòn (do ion Cl⁻, CO₂, môi trường xâm thực), sản phẩm gỉ có thể tích gấp 2–6 lần thép ban đầu, tạo áp lực giãn nở lên lớp bê tông bảo vệ. 
Khi ứng suất này vượt cường độ kéo của bê tông bảo vệ, lớp bê tông bị nứt, bong tách, tạo đường cho tác nhân ăn mòn xâm nhập sâu hơn.""",
            "Đặc trưng hình dạng / hình học": """
Vết nứt thường chạy dọc theo thanh cốt thép, có thể chỉ là vết nứt dưới bề mặt rồi lan ra ngoài. 
Có hiện tượng đổi màu bề mặt (vệt gỉ), bong lớp bê tông bảo vệ, lộ thép gỉ.""",
            "Thời gian xuất hiện": "Thường sau vài năm (≥ 2 năm) tùy điều kiện môi trường và lớp bảo vệ.",
            "Cách kiểm soát / phòng ngừa": """
- Đảm bảo chiều dày và chất lượng lớp bê tông bảo vệ, bê tông đặc chắc, ít thấm. 
- Dùng cốt thép chống ăn mòn (thép mạ, thép không gỉ) khi cần.
- Dùng phụ gia ức chế ăn mòn, phụ gia giảm thấm.
- Kiểm soát clo, CO₂, nước thấm; bảo trì, sửa chữa kịp thời các vết nứt và bong tróc bề mặt."""
        },
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do tải trọng – uốn",
            "Nguyên nhân hình thành": """
Do mô men uốn hoặc ứng suất kéo vượt quá khả năng chịu kéo của bê tông tại vùng chịu kéo của dầm, sàn, bản. 
Thiết kế thiếu cốt thép chịu uốn, tiết diện không đủ, hoặc tải trọng sử dụng vượt tải thiết kế.""",
            "Đặc trưng hình dạng / hình học": """
Vết nứt thường xuất hiện tại vùng kéo, gần giữa nhịp, có xu hướng gần vuông góc với trục cấu kiện. 
Hình dạng đường chéo hoặc hơi cong, rộng nhất ở vùng kéo (dưới bản/dầm) và nhỏ dần về phía vùng nén.""",
            "Thời gian xuất hiện": "Có thể từ vài tháng đến vài năm khi công trình chịu tải lâu dài hoặc quá tải.",
            "Cách kiểm soát / phòng ngừa": """
- Thiết kế đúng tiêu chuẩn, bố trí đủ và đúng vị trí cốt thép chịu uốn. 
- Kiểm soát tải trọng sử dụng, tránh quá tải, tải tập trung không tính trước.
- Khi đã nứt: đánh giá khả năng chịu lực, có thể bơm keo epoxy, gia cường bằng FRP, dầm thép, bản tăng cường..."""
        },
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do tải trọng – cắt / nén / xoắn",
            "Nguyên nhân hình thành": """
- Cắt: khả năng chịu cắt của bê tông/cốt thép đai không đủ, lực cắt lớn gần gối hoặc vùng tải tập trung. 
- Nén: bê tông chịu nén vượt giới hạn, đặc biệt trong cột, tường chịu nén. 
- Xoắn: mô men xoắn lớn khi dầm/sàn chịu tải lệch tâm, không đối xứng, tiết diện và cốt xoắn không đủ.""",
            "Đặc trưng hình dạng / hình học": """
- Nứt cắt: vết nứt xiên ~45° so với trục dầm/sàn, thường gần gối. 
- Nứt nén: vết nứt song song với phương nén, phần giữa vết nứt rộng hơn hai đầu. 
- Nứt xoắn: vết nứt chéo xoắn ốc, dạng ziczac bao quanh cấu kiện, bề rộng gần như đều theo chiều dài.""",
            "Thời gian xuất hiện": "Thường từ vài tháng đến vài năm, phụ thuộc mức độ tải trọng và điều kiện sử dụng.",
            "Cách kiểm soát / phòng ngừa": """
- Tăng cường cốt đai chịu cắt, cốt xoắn, thiết kế hợp lý vùng gối và gối – dầm. 
- Giám sát tải trọng, tránh tải tập trung đột ngột. 
- Đối với nứt nén: kiểm tra lại khả năng chịu lực của cột/tường, có thể gia cường bằng bọc thép, FRP, tăng tiết diện."""
        },
        {
            "Nhóm": "II. Sau khi đông cứng",
            "Loại vết nứt": "Nứt do lún (settlement)",
            "Nguyên nhân hình thành": """
Lún, lún lệch của nền, móng, hoặc rửa trôi lớp đệm, gây biến dạng lớn cho kết cấu bê tông phía trên. 
Sự khác biệt chuyển vị (độ cong lớn của đường cong lún) tạo ứng suất kéo trong dầm, sàn, tường và gây nứt.""",
            "Đặc trưng hình dạng / hình học": """
Vết nứt thường vuông góc với phương ứng suất kéo chính do lún nền. 
Ở dầm, sàn: vết nứt thẳng góc với trục dầm/sàn; lún lệch có thể xuất hiện vết nứt xiên gần liên kết dầm–cột, các vết nứt chéo 45° ở góc sàn.""",
            "Thời gian xuất hiện": "Thường xuất hiện khi tải trọng tác dụng làm nền/móng bắt đầu lún rõ rệt (từ vài tháng đến vài năm).",
            "Cách kiểm soát / phòng ngừa": """
- Thiết kế, xử lý nền móng phù hợp điều kiện địa chất, tránh lún lệch lớn. 
- Kiểm soát tải trọng, tránh thay đổi đột ngột so với thiết kế. 
- Khi đã nứt: đánh giá lún, có thể gia cố nền, móng, gia cường kết cấu, bơm keo epoxy vào vết nứt nếu còn đảm bảo an toàn."""
        },
    ]

    df_detail = pd.DataFrame(detailed_cracks)

    st.dataframe(df_detail, use_container_width=True, height=500)

    # =========================================================
    # 3. CHỌN 1 LOẠI VẾT NỨT ĐỂ XEM CHI TIẾT (UI THÂN THIỆN HƠN)
    # =========================================================
    st.markdown("### 🔍 Tra cứu chi tiết từng loại vết nứt")

    options = [
        f"{row['Nhóm']} – {row['Loại vết nứt']}"
        for row in detailed_cracks
    ]
    selected_label = st.selectbox("Chọn loại vết nứt:", options)

    selected_idx = options.index(selected_label)
    selected = detailed_cracks[selected_idx]

    with st.expander("Chi tiết loại vết nứt đã chọn", expanded=True):
        st.markdown(f"**Nhóm:** {selected['Nhóm']}")
        st.markdown(f"**Loại vết nứt:** {selected['Loại vết nứt']}")
        st.markdown("**Nguyên nhân hình thành:**")
        st.markdown(selected["Nguyên nhân hình thành"])
        st.markdown("**Đặc trưng hình dạng / hình học:**")
        st.markdown(selected["Đặc trưng hình dạng / hình học"])
        st.markdown(f"**Thời gian xuất hiện (điển hình):** {selected['Thời gian xuất hiện']}")
        st.markdown("**Cách kiểm soát / phòng ngừa:**")
        st.markdown(selected["Cách kiểm soát / phòng ngừa"])


# =========================================================
# 4. GIAO DIỆN STREAMLIT
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
    st.caption("Phân biệt ảnh nứt / không nứt (Stage 1).")

st.write("---")

st.sidebar.header("Cấu hình phân tích")
min_conf = st.sidebar.slider(
    "Ngưỡng confidence tối thiểu",
    0.0,
    1.0,
    0.3,
    0.05,
)
st.sidebar.caption("Chỉ hiển thị những vết nứt có độ tin cậy ≥ ngưỡng này.")

# Cho phép tải Nhiều ảnh
uploaded_files = st.file_uploader(
    "Tải một hoặc nhiều ảnh bê tông (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
analyze_btn = st.button("🔍 Phân tích ảnh")

# =========================================================
# 5. XỬ LÝ ẢNH – STAGE 1
# =========================================================

# =========================================================
# 5. XỬ LÝ ẢNH – STAGE 1
# =========================================================

if analyze_btn:
    if not uploaded_files:
        st.warning("Vui lòng chọn ít nhất một ảnh trước khi bấm **Phân tích**.")
        st.stop()

    # Lặp qua từng ảnh
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        st.write("---")
        st.markdown(f"## Ảnh {idx}: `{uploaded_file.name}`")

        t0 = time.time()
        orig_img = Image.open(uploaded_file).convert("RGB")
        img_w, img_h = orig_img.size

        # Gửi tới Roboflow
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

        # ---------------- 2 cột: Ảnh gốc – Ảnh phân tích ----------------
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

        # Nếu không có vết nứt thì bỏ qua phần báo cáo chi tiết
        if len(preds_conf) == 0 or analyzed_img is None:
            continue

        # =====================================================
        # 5.1. BÁO CÁO CHI TIẾT + STAGE 2 Ở TAB RIÊNG
        # =====================================================
        st.write("---")
        tab_stage1, tab_stage2 = st.tabs(
            [
                "Stage 1 – Báo cáo chi tiết",
                "Stage 2 – Phân loại vết nứt",
            ]
        )

        with tab_stage1:
            st.subheader("Bảng thông tin vết nứt")

            confs = [float(p.get("confidence", 0)) for p in preds_conf]
            avg_conf = sum(confs) / len(confs)
            map_val = round(min(1.0, avg_conf - 0.05), 2)

            # Tính % diện tích vùng nứt lớn nhất
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

            # ---------- BIỂU ĐỒ ----------
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
                labels = ["Vùng nứt lớn nhất", "Phần ảnh còn lại"]
                sizes = [max_ratio, 1 - max_ratio]
                plt.figure(figsize=(4, 3))
                plt.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=140,
                    colors=["#ef5350", "#90caf9"],
                )
                plt.title("Tỷ lệ vùng nứt so với toàn ảnh")
                st.pyplot(plt.gcf())
                plt.close()

            # ---------- NÚT TẢI PDF ----------
            pdf_buf = export_pdf(orig_img, analyzed_img, metrics_df)
            st.download_button(
                "📄 Tải báo cáo PDF cho ảnh này",
                data=pdf_buf,
                file_name=f"BKAI_CrackReport_{uploaded_file.name.split('.')[0]}.pdf",
                mime="application/pdf",
                key=f"pdf_btn_{idx}_{uploaded_file.name}",
            )

        with tab_stage2:
            show_stage2_demo()








