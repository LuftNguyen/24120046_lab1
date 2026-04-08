import requests
import json

# CHÚ Ý: Thay đoạn URL này bằng link Pinggy thực tế của bạn
API_URL = "https://lzoga-35-238-127-76.run.pinggy-free.link/predict"

# Giả lập người dùng gửi lên 5 điểm cần giao hàng
payload = {
    "start_location": {"id": 0, "x": 0.8, "y": 0.2}, # User đang ở giữa bản đồ
    "destinations": [
        {"id": 1, "x": 0.1, "y": 0.1},
        {"id": 2, "x": 0.9, "y": 0.4},
        {"id": 3, "x": 0.8, "y": 0.5},
        {"id": 4, "x": 0.9, "y": 0.1},
        {"id": 5, "x": 0.5, "y": 0.4}
    ]
}

print(f"Đang gửi dữ liệu tới: {API_URL}")
try:
    response = requests.post(API_URL, json=payload)
    
    # In mã trạng thái (200 là thành công)
    print(f"Status Code: {response.status_code}\n")
    
    # In kết quả JSON được format đẹp mắt
    print("Kết quả trả về từ AI:")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Lỗi kết nối: {e}")