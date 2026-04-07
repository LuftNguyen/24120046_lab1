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

app = FastAPI(title="Thực tế hóa TSP API")

# 1. Định nghĩa cấu trúc dữ liệu
class Location(BaseModel):
    id: int
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

class PredictRequest(BaseModel):
    # Điểm User đang đứng
    start_location: Location 
    # Danh sách các điểm cần đến giao hàng/thăm quan
    destinations: List[Location] = Field(..., min_items=1)

# Hàm tính tổng quãng đường của đường đi mở (không tính lượt quay về)
def calculate_open_path_distance(indices, points):
    dist = 0.0
    for i in range(len(indices) - 1):
        p1 = points[indices[i]]
        p2 = points[indices[i+1]]
        dist += math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    return dist

# 2. Nạp Model
MODEL_PATH = "pretrained/epoch-99.pt"
try:
    model, _ = load_model(MODEL_PATH)
    model.eval()
    MODEL_READY = True
except Exception as e:
    print(f"Lỗi khởi tạo model: {e}")
    MODEL_READY = False

@app.post("/predict")
async def predict(request: PredictRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    # BƯỚC 1: Hợp nhất dữ liệu. 
    # Ta luôn coi start_location là điểm đầu tiên (index 0)
    all_points = [request.start_location] + request.destinations
    n_nodes = len(all_points)
    
    # BƯỚC 2: Chuyển thành Tensor [1, N, 2]
    coords = torch.tensor([[[p.x, p.y] for p in all_points]], dtype=torch.float32)

    try:
        with torch.no_grad():
            model.set_decode_type("greedy")
            # AI trả về: chi phí, xác suất, và lộ trình (pi)
            cost, _, pi = model(coords, return_pi=True)
        
        # pi là mảng các index, ví dụ: [2, 0, 3, 1]
        tour_indices = pi[0].cpu().numpy().tolist()

        # BƯỚC 3: "XOAY MẢNG" ĐỂ ĐƯA ĐIỂM BẮT ĐẦU LÊN ĐẦU VÀ SO SÁNH
        start_position_in_tour = tour_indices.index(0)
        
        # 3.1: Tạo mảng Chiều Xuôi
        forward_indices = tour_indices[start_position_in_tour:] + tour_indices[:start_position_in_tour]
        
        # 3.2: Tạo mảng Chiều Ngược (Giữ nguyên số 0 ở đầu, đảo chiều toàn bộ khúc sau)
        reverse_indices = [forward_indices[0]] + forward_indices[1:][::-1]

        # 3.3: So sánh khoảng cách đường thẳng (không quay về nhà)
        dist_forward = calculate_open_path_distance(forward_indices, all_points)
        dist_reverse = calculate_open_path_distance(reverse_indices, all_points)

        # 3.4: Chọn lộ trình tối ưu nhất cho người giao hàng
        if dist_reverse < dist_forward:
            final_indices = reverse_indices
            final_distance = dist_reverse
        else:
            final_indices = forward_indices
            final_distance = dist_forward

        # BƯỚC 4: Trả về kết quả đã sắp xếp
        optimized_order = [all_points[i].model_dump() for i in final_indices]

        return {
            "status": "success",
            "start_point_id": request.start_location.id,
            "total_distance": round(final_distance, 4), # Trả về biến final_distance mới tính toán
            "optimized_route": optimized_order
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))