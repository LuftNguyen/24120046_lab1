# AI Routing Optimization Service (Open TSP)

## Thông tin sinh viên
- **Họ và tên:** Nguyễn Hà Minh Hiền
- **MSSV:** 24120046
- **Trường:** Đại học Khoa Học Tự Nhiên TP.HCM - Khoa Công Nghệ Thông Tin
- **Môn học:** Tư Duy Tính Toán
- **Lớp**: 24CTT5

## Tên mô hình và Liên kết
- **Tên mô hình:** Attention Learn to Route (dựa trên kiến trúc của Wouter Kool).
- **Github Repository:** [https://github.com/wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) 


##  Mô tả ngắn về chức năng của hệ thống.
Hệ thống cung cấp một RESTful API được xây dựng bằng FastAPI để giải quyết bài toán Người giao hàng (Traveling Salesperson Problem - TSP)

Người dùng có thể gửi lên tọa độ GPS thực tế (Kinh độ, Vĩ độ) của vị trí hiện tại và danh sách các điểm cần đến. Hệ thống sẽ tự động tiền xử lý (chuẩn hóa tỷ lệ bản đồ về (0.0 - 1.0)), gọi mô hình Deep Learning để suy luận, và trả về một lộ trình di chuyển (thứ tự địa điểm đã được tối ưu) giúp người dùng di chuyển thuận tiện hơn.

## Hướng dẫn cài đặt thư viện
Yêu cầu hệ thống đã cài đặt sẵn Python (phiên bản 3.8 trở lên). Mở Terminal/Command Prompt tại thư mục gốc của dự án và chạy lệnh sau để cài đặt toàn bộ thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Hướng dẫn chạy chương trình

Hệ thống cung cấp hai cách để triển khai: chạy trực tiếp trên máy cá nhân hoặc mở kết nối công khai qua Pinggy.

### Cách 1: Chạy trên máy tính cá nhân (Run Locally)
Sau khi cài đặt xong các thư viện cần thiết, bạn khởi động máy chủ FastAPI bằng lệnh sau tại thư mục gốc của dự án:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Cách 2: Đưa API lên Internet bằng Pinggy
Bạn có thể dùng Pinggy để chia sẻ API cho người khác
Đảm bảo máy chủ cục bộ đang chạy và mở Terminal/Command Prompt mới và chạy lệnh SSH

```bash
ssh -p 443 -R0:localhost:8000 a.pinggy.io
```
## Hướng dẫn gọi API và  ví dụ request/response

### 1. Lấy thông tin giới thiệu hệ thống
- **Endpoint:**  `GET /`
- **Mô tả:** Trả về thông tin giới thiệu ngắn gọn về hệ thống.
- **Ví dụ Request:** *(Không yêu cầu Body)*
- **Ví dụ Response:**
```json
{
  "system_name": "AI Routing Optimization Service",
  "description": "API nhận tọa độ GPS bất kỳ, sử dụng AI để tìm lộ trình giao hàng ngắn nhất không quay về điểm xuất phát.",
  "author": "Nguyen Ha Minh Hien"
}
```

### 2. Kiểm tra trạng thái hệ thống
- **Endpoint:** `GET /health`
- **Mô tả:** Kiểm tra xem máy chủ và mô hình AI đã nạp thành công và sẵn sàng hoạt động chưa.
- **Ví dụ Request:** *(Không yêu cầu Body)*
- **Ví dụ Response:**
```json
{
  "status": "ok",
  "model_state": "loaded and ready"
}
```

### 3. Lấy lộ trình tối ưu
- **Endpoint:** `POST /predict`
- **Mô tả:** Gửi vị trí bắt đầu và danh sách các điểm đến để nhận về lộ trình di chuyển ngắn nhất.
- **Headers:** `Content-Type: application/json`

- **Ví dụ Request:**
```json
{
  "start_location": {
    "id": 0,
    "x": 0.8,
    "y": 0.2
  },
  "destinations": [
    { "id": 1, "x": 0.1, "y": 0.1 },
    { "id": 2, "x": 0.9, "y": 0.4 },
    { "id": 3, "x": 0.8, "y": 0.5 },
    { "id": 4, "x": 0.9, "y": 0.1 },
    { "id": 5, "x": 0.5, "y": 0.4 }
  ]
}
```

- **Ví dụ Response:**

```json
{
  "status": "success",
  "start_point_id": 0,
  "total_locations": 6,
  "total_distance": 1.3984,
  "optimized_route": [
    {
      "id": 0,
      "x": 0.8,
      "y": 0.2
    },
    {
      "id": 4,
      "x": 0.9,
      "y": 0.1
    },
    {
      "id": 2,
      "x": 0.9,
      "y": 0.4
    },
    {
      "id": 3,
      "x": 0.8,
      "y": 0.5
    },
    {
      "id": 5,
      "x": 0.5,
      "y": 0.4
    },
    {
      "id": 1,
      "x": 0.1,
      "y": 0.1
    }
  ]
}
```





