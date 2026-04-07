import torch
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

# 2. Nạp Model
MODEL_PATH = "pretrained/epoch-99.pt"
try:
    model, _ = load_model(MODEL_PATH)
    model.eval()
    MODEL_READY = True
except:
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

        # BƯỚC 3: "XOAY MẢNG" ĐỂ ĐƯA ĐIỂM BẮT ĐẦU LÊN ĐẦU
        # Vì all_points[0] là start_location, ta tìm xem số 0 nằm ở đâu trong tour_indices
        start_position_in_tour = tour_indices.index(0)
        
        # Xoay mảng sao cho số 0 đứng đầu
        # Ví dụ: [2, 0, 3, 1] -> [0, 3, 1, 2]
        final_indices = tour_indices[start_position_in_tour:] + tour_indices[:start_position_in_tour]

        # BƯỚC 4: Trả về kết quả đã sắp xếp
        optimized_order = [all_points[i].model_dump() for i in final_indices]

        return {
            "status": "success",
            "start_point_id": request.start_location.id,
            "total_distance": float(cost[0]), # Tổng quãng đường AI tính toán
            "optimized_route": optimized_order
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))