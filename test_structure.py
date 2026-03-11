import torch
from detectron2.config import get_cfg
from diffusiondet import add_diffusiondet_config
from diffusiondet.detector import DiffusionDet

def test_model_structure():
    print("--- Đang khởi tạo cấu hình ---")
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    # Giả lập cấu hình tối thiểu
    cfg.MODEL.DEVICE = "cpu"
    
    print("--- Đang khởi tạo mô hình DiffusionDet ---")
    # Khởi tạo model lần đầu để lấy thông tin backbone
    try:
        model = DiffusionDet(cfg)
        
        # Tự động lấy các tính năng mà backbone thực sự tạo ra
        actual_features = list(model.backbone.output_shape().keys())
        print(f"Backbone tạo ra các features: {actual_features}")
        cfg.MODEL.ROI_HEADS.IN_FEATURES = actual_features
        
        # Khởi tạo lại model với config đã chuẩn hóa features
        model = DiffusionDet(cfg)
    except Exception as e:
        print(f"Lỗi khi khởi tạo model: {e}")
        # Fallback nếu không tự lấy được
        actual_features = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.ROI_HEADS.IN_FEATURES = actual_features
        model = DiffusionDet(cfg)

    model.eval()

    print("\n--- Kiểm tra cấu trúc DynamicHead ---")
    head = model.head
    # Kiểm tra xem các layer phân cấp có tồn tại không
    first_head = head.head_series[0]
    has_jaw_to_tooth = hasattr(first_head, "jaw_to_tooth")
    has_tooth_jaw_to_patho = hasattr(first_head, "tooth_jaw_to_patho")
    
    print(f"Có layer jaw_to_tooth: {has_jaw_to_tooth}")
    print(f"Có layer tooth_jaw_to_patho: {has_tooth_jaw_to_patho}")

    if has_jaw_to_tooth and has_tooth_jaw_to_patho:
        print("=> Cấu trúc phân cấp đã được thiết lập đúng trong RCNNHead.")
    else:
        print("=> LỖI: Thiếu các layer phân cấp!")

    print("\n--- Kiểm tra Forward Pass (Dummy Input) ---")
    # Giả lập input (1 ảnh 224x224)
    dummy_image = torch.randn(3, 224, 224)
    batched_inputs = [{"image": dummy_image, "height": 224, "width": 224}]
    
    try:
        with torch.no_grad():
            # Trong chế độ eval, model trả về results (processed instances)
            results = model(batched_inputs)
        
        print(f"Số lượng instance dự đoán: {len(results[0]['instances'])}")
        instances = results[0]['instances']
        
        # Kiểm tra xem Instances có chứa các trường mới không
        has_pred_patho = hasattr(instances, "pred_patho")
        has_pred_jaw = hasattr(instances, "pred_jaw")
        
        print(f"Kết quả có pred_patho: {has_pred_patho}")
        print(f"Kết quả có pred_jaw: {has_pred_jaw}")
        
        if has_pred_patho and has_pred_jaw:
            print("=> Forward pass thành công và trả về đầy đủ các trường đa nhãn.")
        else:
            print("=> CẢNH BÁO: Kết quả thiếu các trường đa nhãn, hãy kiểm tra lại logic inference.")
            
    except Exception as e:
        print(f"=> LỖI khi chạy forward pass: {e}")

if __name__ == "__main__":
    test_model_structure()
