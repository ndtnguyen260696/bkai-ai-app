    st.write("---")

    # ------------ FORM THÔNG TIN NGƯỜI DÙNG (BẮT BUỘC) ------------
    if "profile_filled" not in st.session_state:
        st.session_state.profile_filled = False

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
                # Lưu vào session_state để dùng cho lần chạy hiện tại
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

        # Nếu chưa fill form đúng, dừng tại đây, chưa cho upload ảnh
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
