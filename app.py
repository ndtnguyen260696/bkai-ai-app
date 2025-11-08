import streamlit as st
import pandas as pd
import requests
from PIL import Image, ImageDraw
import io
import matplotlib.pyplot as plt
import time

# ==============================
# 1️⃣ CẤU HÌNH CHUNG
# ==============================
st.set_page_config(page_title="BKAI – Concrete Crack Detection", layout="wide")

# CSS giao diện giống website BKAI
st.markdown("""
<style>
body { background-color: #f8fafc; color: #1e293b; }
h1,h2,h3,h4,h5 { color:#0f172a; text-align:center; font-family: 'Segoe UI'; }
header, footer {visibility: hidden;}
[data-testid="stSidebar"] {background-color: #f1f5f9;}
div.block-container {padding-top: 1rem;}
.bkai-title {text-align:center; color:#0f172a; font-weight:bold; font-size:28px;}
</style>
""", unsafe_allow_html=True)

# ==============================
# 2️⃣ LOGO VÀ HEADER
# ==============================
col_logo, col_title = st.columns([1,4])
with col_logo:
    st.image("bkai_logo.png", width=120)
with col_title:
    st.markdown("<h1 class='bkai-title'>BKAI – AI Concrete Crack Inspection Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Ứng dụng AI phát hiện và phân loại vết nứt bê tông – Powered by BKAI</p>", unsafe_allow_html=True)
st.divider()

# ==============================
# 3️⃣ TRANG ĐĂNG NHẬP
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("🔐 Đăng nhập để sử dụng hệ thống")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Mật khẩu", type="password")
        submit = st.form_submit_button("Đăng nhập")

        if submit:
            # 👉 DEMO: Cho phép mọi email hợp lệ đăng nhập
            if "@" in email and len(password) >= 3:
                st.session_state.logged_in = True
                st.success("✅ Đăng nhập thành công!")
                st.experimental_rerun()
            else:
                st.error("❌ Sai thông tin đăng nhập.")
    st.stop()

# ==============================
# 4️⃣ TRANG PHÂN TÍCH ẢNH
# ==============================
st.success(f"Xin chào **{email}**, hãy tải ảnh để hệ thống phân tích 🔍")

# Link mô hình Roboflow CNN
ROBOFLOW_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"

# Upload nhiều ảnh (tối đa 20)
uploaded_files = st.file_uploader(
    "📂 Tải lên ảnh bê tông cần phân tích (1–20 ảnh)",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    help="Bạn có thể chọn nhiều ảnh cùng lúc để phân tích song song."
)

if uploaded_files:
    for idx, file in enumerate(uploaded_files, start=1):
        st.divider()
        st.markdown(f"### 🖼️ Ảnh {idx}: `{file.name}`")

        # Đọc ảnh
        image = Image.open(file).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        # ===============================
        # GỬI YÊU CẦU TỚI ROBOFLOW
        # ===============================
        with st.spinner("⏳ Đang phân tích ảnh bằng mô hình CNN..."):
            t0 = time.time()
            try:
                resp = requests.post(ROBOFLOW_URL, files={"file": ("image.jpg", img_bytes, "image/jpeg")})
                latency = time.time() - t0
                data = resp.json()
            except Exception as e:
                st.error(f"Lỗi khi gọi API Roboflow: {e}")
                continue

        preds = data.get("predictions", [])
        conf_thresh = 0.3
        preds = [p for p in preds if p["confidence"] >= conf_thresh]

        # ===============================
        # HIỂN THỊ ẢNH
        # ===============================
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc / Original Image", use_column_width=True)

        # Vẽ box và label
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for p in preds:
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            x0, y0, x1, y1 = x - w/2, y - h/2, x + w/2, y + h/2
            label = f"{p['class']} {p['confidence']:.2f}"
            draw.rectangle([x0, y0, x1, y1], outline="green", width=3)
            draw.text((x0, y0-12), label, fill="black")

        with col2:
            st.image(annotated, caption="Ảnh đã phân tích / Analyzed Image", use_column_width=True)

        # ===============================
        # KẾT LUẬN CHUNG
        # ===============================
        if preds:
            st.error("⚠️ Có vết nứt được phát hiện!")
        else:
            st.success("✅ Không phát hiện vết nứt rõ ràng.")

        # ===============================
        # BẢNG THỐNG KÊ KẾT QUẢ
        # ===============================
        total_cracks = len(preds)
        avg_conf = sum(p["confidence"] for p in preds)/total_cracks if total_cracks>0 else 0

        df = pd.DataFrame(
            {
                "Thông số / Parameter": [
                    "Số vùng nứt / Crack regions",
                    "Độ tin cậy TB / Avg confidence",
                    "Ngưỡng phát hiện / Threshold",
                    "Thời gian xử lý / Inference time (s)",
                    "Kết luận / Conclusion"
                ],
                "Giá trị / Value": [
                    total_cracks,
                    f"{avg_conf:.2f}",
                    f"{conf_thresh:.2f}",
                    f"{latency:.2f}",
                    "Có vết nứt / Crack detected" if preds else "Không có / None"
                ]
            }
        )
        st.subheader("📊 Báo cáo chi tiết / Crack Analysis Summary")
        st.table(df)

        # ===============================
        # BIỂU ĐỒ MINH HỌA
        # ===============================
        st.subheader("📈 Biểu đồ minh họa / Visual Charts")

        if preds:
            confs = [p["confidence"] for p in preds]
            widths = [p["width"] for p in preds]
            heights = [p["height"] for p in preds]

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                # Biểu đồ bar độ tin cậy
                fig, ax = plt.subplots()
                ax.bar(range(len(confs)), confs, color="#0ea5e9")
                ax.set_title("Confidence per crack")
                ax.set_xlabel("Crack #")
                ax.set_ylabel("Confidence")
                st.pyplot(fig)

            with col_b:
                # Pie chart tỷ lệ có/không nứt
                fig2, ax2 = plt.subplots()
                ax2.pie([len(preds), 20-len(preds)], labels=["Crack", "No Crack"],
                        autopct="%1.0f%%", colors=["#ef4444", "#22c55e"])
                ax2.set_title("Crack Presence Ratio")
                st.pyplot(fig2)

            with col_c:
                # Scatter chiều rộng – chiều cao
                fig3, ax3 = plt.subplots()
                ax3.scatter(widths, heights, c=confs, cmap="plasma", s=80)
                ax3.set_xlabel("Width (px)")
                ax3.set_ylabel("Height (px)")
                ax3.set_title("Crack Size Distribution")
                st.pyplot(fig3)

        else:
            st.info("Không có vết nứt để hiển thị biểu đồ.")

else:
    st.info("⬆️ Hãy đăng nhập và tải lên ảnh để bắt đầu phân tích.")
