import torch
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Giúp Python tìm thấy các thư mục 'nets' và 'utils'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model

# --- 1. Khởi tạo FastAPI Application ---
app = FastAPI(
    title="TSP Optimization API", 
    description="API giải quyết bài toán Người giao hàng (TSP) bằng Attention Model với số lượng điểm biến thiên."
)

# --- 2. Định nghĩa cấu trúc dữ liệu đầu vào (Pydantic Models) ---
# Yêu cầu: "Cần có kiểm tra dữ liệu đầu vào ở mức cơ bản"
class Location(BaseModel):
    id: int = Field(..., description="ID của địa điểm")
    x: float = Field(..., ge=0.0, le=1.0, description="Tọa độ X (chuẩn hóa từ 0 đến 1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Tọa độ Y (chuẩn hóa từ 0 đến 1)")

class PredictRequest(BaseModel):
    # Yêu cầu: Ít nhất 3 điểm mới có thể tối ưu lộ trình
    locations: List[Location] = Field(..., min_items=3, description="Danh sách các địa điểm cần đi qua (tối thiểu 3)")

# --- 3. Nạp Model khi khởi động ---
MODEL_PATH = "pretrained/epoch-99.pt" # Đảm bảo file này tồn tại
try:
    model, _ = load_model(MODEL_PATH)
    model.eval()
    MODEL_READY = True
except Exception as e:
    print(f"Lỗi nạp mô hình: {e}")
    MODEL_READY = False

# --- 4. CÁC ĐẦU MÚT API (ENDPOINTS) ---

# Yêu cầu: GET / (Thông tin giới thiệu ngắn gọn)
@app.get("/")
async def root():
    return {
        "system_name": "AI Routing Optimization Service",
        "description": "Hệ thống nhận danh sách tọa độ biến thiên và trả về thứ tự di chuyển tối ưu nhất bằng trí tuệ nhân tạo.",
        "endpoints": ["/health", "/predict"]
    }

# Yêu cầu: GET /health (Kiểm tra trạng thái hệ thống)
@app.get("/health")
async def health_check():
    if MODEL_READY:
        return {"status": "ok", "model_state": "loaded and ready"}
    else:
        # Xử lý lỗi nếu model chưa sẵn sàng
        raise HTTPException(status_code=503, detail="Service Unavailable: Mô hình chưa được nạp thành công.")

# Yêu cầu: POST /predict (Nhận dữ liệu, gọi mô hình, trả kết quả JSON)
@app.post("/predict")
async def predict_route(request: PredictRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Service Unavailable: Hệ thống AI đang bảo trì.")

    locations = request.locations
    n_nodes = len(locations)

    try:
        # Chuẩn bị Tensor đầu vào từ danh sách tọa độ (Input Tensor)
        coords = torch.tensor([[[loc.x, loc.y] for loc in locations]], dtype=torch.float32)
        
        with torch.no_grad():
            # 1. Cài đặt chiến lược giải (Greedy: chọn điểm tốt nhất ở mỗi bước)
            model.set_decode_type("greedy")
            
            # 2. Suy luận và yêu cầu trả về thứ tự điểm (return_pi=True)
            cost, _, pi = model(coords, return_pi=True)
        
        # Lấy thứ tự index
        tour_indices = pi[0].cpu().numpy().tolist()
        
        # Yêu cầu: "Kết quả trả về phải rõ ràng, có cấu trúc"
        optimized_order = [locations[i].model_dump() for i in tour_indices]
        
        return {
            "status": "success",
            "message": "Lộ trình đã được tối ưu hóa thành công.",
            "total_locations": n_nodes,
            "optimized_order": optimized_order
        }

    # Yêu cầu: "Xử lý hợp lý các trường hợp lỗi trong quá trình suy luận"
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=f"Lỗi phần cứng/Tensor trong quá trình suy luận: {str(re)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống không xác định: {str(e)}")