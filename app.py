# app.py
import streamlit as st
import requests
from PIL import Image, ImageOps
import io
import base64
import datetime

# ===============================
# CẤU HÌNH: dán URL detect.roboflow.com từ Deploy -> Hosted API -> Python
# Ví dụ: "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=XXXXXXXX"
# ===============================
ROBOFLOW_FULL_URL = "https://detect.roboflow.com/crack_segmentation_detection/4?api_key=nWA6ayjI5bGNpXkkbsAb"

# ===============================
# Hàm phụ: overlay mask (hỗ trợ 2 dạng: base64 data URL hoặc URL của ảnh mask)
# Nếu response chứa mask per-prediction (ví dụ "mask": "data:image/png;base64,....")
# hoặc "mask": "https://.../mask.png" thì hàm sẽ overlay lên ảnh gốc.
# ===============================
def overlay_mask_on_image(image: Image.Image, mask_image: Image.Image, alpha: float = 0.45):
    """
    image: PIL image RGB
    mask_image: PIL image (grayscale or RGBA) same size (or will be resized)
    alpha: opacity of mask overlay
    """
    # Resize mask to match main image if needed
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, resample=Image.NEAREST)

    # Ensure mask is RGBA
    mask_rgba = mask_image.convert("RGBA")

    # Colorize mask (yellow) — convert grayscale to single color
    color = (255, 215, 0, int(255 * alpha))  # gold/yellow with alpha
    colored_mask = Image.new("RGBA", mask_rgba.size)
    # Use mask as alpha
    mask_alpha = mask_rgba.split()[-1] if mask_rgba.mode == "RGBA" else mask_rgba.convert("L")
    colored_mask.paste(color, (0, 0), mask=mask_alpha)

    base = image.convert("RGBA")
    combined = Image.alpha_composite(base, colored_mask)
    return combined.convert("RGB")


def mask_from_data(data_str):
    """ data_str may be 'data:image/png;base64,...' or a direct base64 string """
    if data_str.startswith("data:"):
        # format: data:image/png;base64,<base64>
        _, b64 = data_str.split(",", 1)
    else:
        b64 = data_str
    b = base64.b64decode(b64)
    return Image.open(io.BytesIO(b))


def mask_from_url(url):
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="BKAI - Crack Segmentation", layout="wide")
st.title("BKAI – Model CNN Phân tích vết nứt bê tông ")
st.write("Upload ảnh, model CNN sẽ trả kết quả ảnh bê tông nứt hay không nứt.")

with st.form("upload_form"):
    name = st.text_input("Họ và tên")
    email = st.text_input("Email")
    note = st.text_area("Ghi chú / Công trình (nếu có)")
    uploaded_file = st.file_uploader("Chọn ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Phân tích ảnh")

if submitted:
    if uploaded_file is None:
        st.warning("Vui lòng chọn ảnh trước khi bấm Phân tích.")
    else:
        # Hiển thị ảnh gốc
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Không thể đọc ảnh: {e}")
            raise

        st.image(image, caption="Ảnh gốc", use_column_width=True)
        st.write("Đang gửi ảnh tới Roboflow... (có thể mất vài giây)")

        # Chuyển ảnh sang bytes JPEG
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        # Gửi request
        try:
            resp = requests.post(
                ROBOFLOW_FULL_URL,
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi khi gọi API: {e}")
        else:
            st.write(f"Status code: {resp.status_code}")
            if resp.status_code != 200:
                st.error("Roboflow trả lỗi — kiểm tra lại URL/API key và quyền truy cập (403 nếu forbidden).")
                # show server text for debugging (cắt ngắn)
                st.text(resp.text[:1500])
            else:
                try:
                    result = resp.json()
                except Exception as e:
                    st.error(f"Không parse được JSON trả về: {e}")
                    st.text(resp.text[:2000])
                else:
                    st.subheader("Kết quả JSON (raw)")
                    st.json(result)

                    # Hiển thị tổng số predictions nếu có
                    preds = result.get("predictions", [])
                    st.write(f"Số prediction: {len(preds)}")

                    # Nếu có mask data trong mỗi prediction -> overlay
                    # Các key có thể khác nhau; ta kiểm tra phổ biến: 'mask', 'segmentation', 'mask_url'
                    overlayed = image.copy()
                    applied_any = False

                    for i, p in enumerate(preds):
                        # nhiều model trả polygon, box, hoặc mask. Thử tìm mask data
                        mask_data = None

                        # kiểm tra một vài key phổ biến
                        for key in ("mask", "segmentation", "mask_url", "mask_image"):
                            if key in p and p[key]:
                                mask_data = p[key]
                                break

                        if mask_data:
                            try:
                                if isinstance(mask_data, str) and mask_data.startswith("http"):
                                    mask_img = mask_from_url(mask_data)
                                else:
                                    # có thể là base64 data URL
                                    mask_img = mask_from_data(mask_data)
                                overlayed = overlay_mask_on_image(overlayed, mask_img, alpha=0.45)
                                applied_any = True
                            except Exception as e:
                                # nếu parse mask fail — không dừng toàn bộ app
                                st.warning(f"Không thể xử lý mask cho prediction #{i}: {e}")

                    if applied_any:
                        st.subheader("Ảnh có overlay mask (nếu mask được trả về từ model)")
                        st.image(overlayed, use_column_width=True)
                    else:
                        st.info("Model không trả mask dưới dạng image URL hoặc base64 trong trường 'predictions'.\nBạn có thể kiểm tra JSON raw ở trên để tìm key chứa segmentation/mask và báo mình.")

                    # Hiển thị thông tin người dùng và log
                    st.write("---")
                    st.write("Thông tin người dùng:")
                    st.write(f"- Tên: {name}")
                    st.write(f"- Email: {email}")
                    st.write(f"- Ghi chú: {note}")
                    st.write(f"- Thời gian: {datetime.datetime.now()}")









