import requests
import json

# CHÚ Ý: Thay đoạn URL này bằng link Pinggy thực tế của bạn
API_URL = "https://jjgsh-35-238-127-76.run.pinggy-free.link/predict"

# Giả lập người dùng gửi lên 5 điểm cần giao hàng
payload = {
    "locations": [
        {"id": 1, "x": 0.10, "y": 0.90},
        {"id": 2, "x": 0.50, "y": 0.50},
        {"id": 3, "x": 0.90, "y": 0.10},
        {"id": 4, "x": 0.20, "y": 0.20},
        {"id": 5, "x": 0.80, "y": 0.80}
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