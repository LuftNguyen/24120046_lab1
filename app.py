import torch
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import sys
import os

# Đảm bảo Python tìm thấy các module trong folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.functions import load_model

# --- KHỞI TẠO FASTAPI ---
app = FastAPI(
    title="TSP Optimization API",
    description="Hệ thống AI tối ưu hóa lộ trình giao hàng thực tế (Hỗ trợ GPS).",
    version="1.0.0"
)

# --- 1. ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU ---
class Location(BaseModel):
    id: int
    x: float = Field(..., description="Tọa độ X hoặc Kinh độ thực tế")
    y: float = Field(..., description="Tọa độ Y hoặc Vĩ độ thực tế")

class PredictRequest(BaseModel):
    start_location: Location = Field(..., description="Vị trí bắt đầu của User")
    destinations: List[Location] = Field(..., min_items=1, description="Các điểm cần đi")

# --- 2. HÀM BỔ TRỢ (TIỀN XỬ LÝ VÀ HẬU XỬ LÝ) ---
def normalize_coordinates(points):
    """Chuẩn hóa GPS thực tế về khoảng 0.0 - 1.0 cho AI"""
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0
    
    normalized_list = []
    for p in points:
        norm_x = (p.x - min_x) / range_x
        norm_y = (p.y - min_y) / range_y
        normalized_list.append([norm_x, norm_y])
        
    return normalized_list

def calculate_open_path_distance(indices, points):
    """Tính tổng quãng đường thực tế (Không tính lượt quay về)"""
    dist = 0.0
    for i in range(len(indices) - 1):
        p1 = points[indices[i]]
        p2 = points[indices[i+1]]
        # Sử dụng khoảng cách Euclid cơ bản
        dist += math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    return dist

# --- 3. NẠP MODEL KHI KHỞI ĐỘNG ---
MODEL_PATH = "pretrained/epoch-99.pt"
try:
    model, _ = load_model(MODEL_PATH)
    model.eval()
    MODEL_READY = True
except Exception as e:
    print(f"Lỗi khởi tạo model: {e}")
    MODEL_READY = False

# --- 4. CÁC ĐẦU MÚT API (ENDPOINTS) ---

@app.get("/")
async def root():
    """Trả về thông tin giới thiệu ngắn gọn về hệ thống"""
    return {
        "system_name": "AI Routing Optimization Service",
        "description": "API nhận tọa độ GPS bất kỳ, sử dụng AI để tìm lộ trình giao hàng ngắn nhất không quay về điểm xuất phát.",
        "author": "Nguyen Ha Minh Hien"
    }

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái hoạt động của hệ thống"""
    if MODEL_READY:
        return {"status": "ok", "model_state": "loaded and ready"}
    else:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model chưa sẵn sàng.")

@app.post("/predict")
async def predict(request: PredictRequest):
    """Nhận dữ liệu, tiền xử lý, gọi AI và trả kết quả JSON"""
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    # BƯỚC 1: Hợp nhất dữ liệu
    all_points = [request.start_location] + request.destinations
    
    # BƯỚC 2: Tiền xử lý (Chuẩn hóa tọa độ) và ép kiểu Tensor
    norm_coords = normalize_coordinates(all_points)
    coords = torch.tensor([norm_coords], dtype=torch.float32)

    try:
        # BƯỚC 3: AI Suy luận
        with torch.no_grad():
            model.set_decode_type("greedy")
            _, _, pi = model(coords, return_pi=True)
        
        tour_indices = pi[0].cpu().numpy().tolist()

        # BƯỚC 4: Hậu xử lý (Xoay mảng & Tìm đường đi mở tốt nhất)
        start_position_in_tour = tour_indices.index(0)
        
        forward_indices = tour_indices[start_position_in_tour:] + tour_indices[:start_position_in_tour]
        reverse_indices = [forward_indices[0]] + forward_indices[1:][::-1]

        # Tính khoảng cách dựa trên TỌA ĐỘ THẬT ban đầu
        dist_forward = calculate_open_path_distance(forward_indices, all_points)
        dist_reverse = calculate_open_path_distance(reverse_indices, all_points)

        if dist_reverse < dist_forward:
            final_indices = reverse_indices
            final_distance = dist_reverse
        else:
            final_indices = forward_indices
            final_distance = dist_forward

        optimized_order = [all_points[i].model_dump() for i in final_indices]

        # BƯỚC 5: Trả về JSON theo cấu trúc rõ ràng
        return {
            "status": "success",
            "start_point_id": request.start_location.id,
            "total_locations": len(all_points),
            "total_distance": round(final_distance, 4),
            "optimized_route": optimized_order
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình suy luận: {str(e)}")