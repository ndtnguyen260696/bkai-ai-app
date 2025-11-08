import csv
import os
import hashlib

# File lưu thông tin người dùng
USER_DB_PATH = "users.csv"


def init_user_db():
    """Tạo file users.csv nếu chưa tồn tại."""
    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "email", "password_hash"])


def _hash_password(password: str) -> str:
    """Hash mật khẩu bằng SHA-256 (không cần cài thêm thư viện)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users():
    """Đọc toàn bộ người dùng từ file CSV."""
    init_user_db()
    users = {}
    with open(USER_DB_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users[row["username"]] = row
    return users


def user_exists(username):
    """Kiểm tra user đã tồn tại chưa."""
    users = load_users()
    return username in users


def register_user(username, email, password):
    """Đăng ký người dùng mới."""
    if user_exists(username):
        return False, "Tên đăng nhập đã tồn tại."

    password_hash = _hash_password(password)

    with open(USER_DB_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([username, email, password_hash])

    return True, "Đăng ký thành công."


def authenticate_user(username, password):
    """Xác thực đăng nhập."""
    users = load_users()
    user = users.get(username)
    if not user:
        return False, "Không tìm thấy tài khoản."

    stored_hash = user["password_hash"]
    if stored_hash == _hash_password(password):
        return True, "Đăng nhập thành công."
    return False, "Mật khẩu không đúng."
