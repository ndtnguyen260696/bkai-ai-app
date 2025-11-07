import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from PIL import Image

# --------------------------------------------------------
# 1. CẤU HÌNH CƠ BẢN
# --------------------------------------------------------
st.set_page_config(page_title="BKAI Crack Report", layout="wide")

# CSS đơn giản cho giống PDF: căn giữa, đường kẻ ngang...
st.markdown(
    """
    <style>
    body { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1, h2, h3, h4 { color: #0f172a; font-family: Arial, sans-serif; }
    table, th, td { font-size: 14px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------
# 2. HÀM VẼ TOÀN BỘ BÁO CÁO LÊN WEB
# --------------------------------------------------------
def render_web_report(
    img_orig: Image.Image,
    img_result: Image.Image,
    df_overview: pd.DataFrame,
    conf_bar_values: dict,
    crack_present_ratio=(1, 0),
):
    """
    Hiển thị giao diện báo cáo giống PDF:
      - Logo + tiêu đề tiếng Việt & Anh
      - Hai ảnh (Ảnh gốc / Ảnh phân tích)
      - Bảng Overview (song ngữ)
      - Biểu đồ Confidence Scores (bar)
      - Biểu đồ Crack Presence (pie)
    """

    # ================= TOP: LOGO + TIÊU ĐỀ =====================
    col_logo, col_title = st.columns([1, 3])

    with col_logo:
        # 👉 Thay 'bkai_logo.png' bằng đường dẫn logo thật của bạn
        try:
            st.image("bkai_logo.png", width=110)
        except Exception:
            st.write("BKAI LOGO")

    with col_title:
        st.markdown(
            """
            <h2 style="text-align:center; margin-bottom:0;">
              BÁO CÁO KIỂM TRA VẾT NỨT BÊ TÔNG
            </h2>
            <h4 style="text-align:center; margin-top:4px; color:#1e293b;">
              Concrete Crack Inspection Report
            </h4>
            """,
            unsafe_allow_html=True,
        )
        today = datetime.date.today().strftime("%B %d, %Y")
        st.markdown(
            f"<p style='text-align:right; font-size:14px;'>{today}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ============ HAI ẢNH: GỐC / PHÂN TÍCH ==============
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h4 style='text-align:center;'>Ảnh gốc</h4>",
            unsafe_allow_html=True,
        )
        st.image(img_orig, use_column_width=True)
        st.markdown(
            "<p style='text-align:center;'>Ảnh gốc / Original Image</p>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            "<h4 style='text-align:center;'>Ảnh phân tích</h4>",
            unsafe_allow_html=True,
        )
        st.image(img_result, use_column_width=True)
        st.markdown(
            "<p style='text-align:center;'>Ảnh phân tích / Result Image</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ================ BẢNG OVERVIEW ======================
    st.markdown(
        "<h3 style='text-align:center;'>Overview</h3>",
        unsafe_allow_html=True,
    )
    st.table(df_overview)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============== 2 BIỂU ĐỒ DƯỚI CÙNG ==================
    col_chart1, col_chart2 = st.columns(2)

    # Biểu đồ bar: Confidence Scores
    with col_chart1:
        st.markdown(
            "<h4 style='text-align:center;'>Confidence Scores</h4>",
            unsafe_allow_html=True,
        )
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        labels = list(conf_bar_values.keys())
        values = list(conf_bar_values.values())
        ax1.bar(labels, values, color="#0ea5e9")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Score")
        plt.xticks(rotation=20)
        st.pyplot(fig1)

    # Biểu đồ pie: Crack Presence
    with col_chart2:
        st.markdown(
            "<h4 style='text-align:center;'>Crack Presence</h4>",
            unsafe_allow_html=True,
        )
        present, absent = crack_present_ratio
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.pie(
            [present, absent],
            labels=["Present", "Absent"],
            autopct="%1.0f%%",
            colors=["#1d4ed8", "#93c5fd"],
        )
        st.pyplot(fig2)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:12px;'>"
        "BKAI © 2025 – Powered by AI for Construction Excellence"
        "</p>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------
# 3. PHẦN MAIN: DEMO + GIẢI THÍCH CẦN THAY Ở ĐÂU
# --------------------------------------------------------
st.sidebar.header("Demo cấu trúc báo cáo")
st.sidebar.write("1. Upload ảnh gốc & ảnh đã phân tích.")
st.sidebar.write("2. App sẽ hiển thị giao diện giống PDF.")
st.sidebar.write("3. Sau này chỉ cần thay số liệu demo bằng kết quả mô hình thật.")

# 👉 Cho phép user upload 2 ảnh để xem layout
orig_file = st.file_uploader("Ảnh gốc / Original Image", type=["jpg", "jpeg", "png"])
result_file = st.file_uploader(
    "Ảnh phân tích / Result Image (có box + mask)", type=["jpg", "jpeg", "png"]
)

if orig_file and result_file:
    img_orig = Image.open(orig_file).convert("RGB")
    img_result = Image.open(result_file).convert("RGB")

    # ----------------------------------------------------
    # 3.1. TẠO BẢNG OVERVIEW DEMO (bạn SẼ THAY CÁC GIÁ TRỊ NÀY)
    # ----------------------------------------------------
    # ► Ở bản thật, các con số dưới đây sẽ được lấy từ model:
    #   - confidence, mAP, detection_score, segmentation_score,
    #   - inference_time_ms, conclusion_text, ...
    confidence_demo = 0.50
    map_demo = 0.48
    detection_demo = 0.35
    segmentation_demo = 0.65
    inference_time_ms_demo = 52
    conclusion_demo = "Có vết nứt / Cracks present in images"

    # Bảng 4 cột giống hình: bên trái & bên phải
    df_overview = pd.DataFrame(
        [
            ["Confidence", f"{confidence_demo:.2f}", "Độ chính xác", f"{confidence_demo:.2f}"],
            ["mAP", f"{map_demo:.2f}", "Segmentation", f"{segmentation_demo:.2f}"],
            ["Detection", f"{detection_demo:.2f}", "Inference Time", f"{inference_time_ms_demo} ms"],
            ["Conclusion", conclusion_demo, "", ""],
        ],
        columns=["Metric (Left)", "Value", "Metric (Right)", "Value "],
    )

    # ----------------------------------------------------
    # 3.2. DỮ LIỆU VẼ BIỂU ĐỒ DEMO
    # ----------------------------------------------------
    # Bar chart: 3 cột như hình: Confidence, mAP, Segmentation
    conf_bar_values = {
        "Confidence": confidence_demo,
        "mAP": map_demo,
        "Segmentation": segmentation_demo,
    }

    # Pie chart: 100% Present (demo). Nếu muốn lấy theo model:
    #   present_ratio = số ảnh/vùng có nứt / tổng
    crack_present_ratio = (1, 0)  # (present, absent)

    # ----------------------------------------------------
    # 3.3. GỌI HÀM VẼ BÁO CÁO
    # ----------------------------------------------------
    render_web_report(
        img_orig=img_orig,
        img_result=img_result,
        df_overview=df_overview,
        conf_bar_values=conf_bar_values,
        crack_present_ratio=crack_present_ratio,
    )

else:
    st.info("⬆️ Hãy upload cả 2 ảnh (gốc & đã phân tích) để xem giao diện báo cáo.")
