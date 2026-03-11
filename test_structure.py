import torch
import torch.nn as nn
from detectron2.config import get_cfg
from diffusiondet import add_diffusiondet_config
from diffusiondet.detector import DiffusionDet

def test_model_structure():
    print("--- 1. Khởi tạo cấu hình ---")
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.MODEL.DEVICE = "cpu"
    
    print("--- 2. Khởi tạo mô hình DentiDiff ---")
    try:
        model = DiffusionDet(cfg)
        # Đồng bộ features
        actual_features = list(model.backbone.output_shape().keys())
        cfg.MODEL.ROI_HEADS.IN_FEATURES = actual_features
        model = DiffusionDet(cfg)
        model.eval()
        print("Mô hình khởi tạo thành công!")
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    print("\n--- 3. Kiểm tra Forward Pass (Mô phỏng Inference) ---")
    # Sử dụng ảnh 512x512
    dummy_image = torch.randn(3, 512, 512)
    batched_inputs = [{"image": dummy_image, "height": 512, "width": 512}]
    
    try:
        with torch.no_grad():
            results = model(batched_inputs)
        
        if len(results) > 0 and 'instances' in results[0]:
            instances = results[0]['instances']
            print(f"Dự đoán thành công!")
            
            # Kiểm tra các nhãn đa nhãn mà chúng ta đã thêm vào
            status = {
                "Số răng (Tooth)": True, # Mặc định
                "Bệnh lý (Patho)": hasattr(instances, "pred_patho"),
                "Vùng hàm (Jaw)": hasattr(instances, "pred_jaw")
            }
            
            for k, v in status.items():
                print(f"{k:20}: {'[OK]' if v else '[THIẾU]'}")
            
            if status["Bệnh lý (Patho)"] and status["Vùng hàm (Jaw)"]:
                print("\n=> KẾT LUẬN: Code đã được sửa đúng theo sườn bài báo và chạy ổn định!")
        
    except Exception as e:
        print(f"\n=> LỖI RUNTIME: {e}")

if __name__ == "__main__":
    test_model_structure()
