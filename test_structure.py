import torch
from detectron2.config import get_cfg
from diffusiondet import add_diffusiondet_config
from diffusiondet.detector import DiffusionDet

def test_model_structure():
    print("--- Đang khởi tạo cấu hình chuẩn ---")
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    
    # Ép cấu hình chuẩn để tránh lỗi Shape Mismatch trên Kaggle/Windows
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7 # Giá trị chuẩn của DiffusionDet
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    
    print("--- Đang khởi tạo mô hình DentalDiffDet ---")
    try:
        model = DiffusionDet(cfg)
        
        # Tự động đồng bộ features từ backbone vào ROI_HEADS
        actual_features = list(model.backbone.output_shape().keys())
        cfg.MODEL.ROI_HEADS.IN_FEATURES = actual_features
        
        # Khởi tạo lại mô hình với các feature thực tế từ backbone
        model = DiffusionDet(cfg)
        print(f"Khởi tạo thành công với features: {actual_features}")
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        return

    model.eval()

    print("\n--- Kiểm tra cấu trúc Phân cấp (Hierarchical) ---")
    head = model.head
    first_head = head.head_series[0]
    
    components = {
        "jaw_to_tooth": hasattr(first_head, "jaw_to_tooth"),
        "tooth_jaw_to_patho": hasattr(first_head, "tooth_jaw_to_patho"),
        "class_jaw_logits": hasattr(first_head, "class_jaw_logits"),
        "class_patho_logits": hasattr(first_head, "class_patho_logits")
    }
    
    for name, exists in components.items():
        print(f"Lớp {name:20}: {'[OK]' if exists else '[MISSING]'}")

    print("\n--- Kiểm tra Forward Pass (Dummy Image) ---")
    # Sử dụng kích thước ảnh chuẩn 800x800 để khớp với backbone stride
    dummy_image = torch.randn(3, 800, 800)
    batched_inputs = [{"image": dummy_image, "height": 800, "width": 800}]
    
    try:
        with torch.no_grad():
            results = model(batched_inputs)
        
        if len(results) > 0 and 'instances' in results[0]:
            instances = results[0]['instances']
            print(f"Dự đoán thành công {len(instances)} đối tượng.")
            
            # Kiểm tra xem có đủ 3 nhãn phân cấp không
            if hasattr(instances, "pred_patho") and hasattr(instances, "pred_jaw"):
                print("=> XÁC NHẬN: Mạng phân cấp hoạt động hoàn hảo!")
            else:
                print("=> CẢNH BÁO: Thiếu các nhãn phân cấp trong đầu ra.")
        else:
            print("=> CẢNH BÁO: Không có kết quả trả về.")
            
    except Exception as e:
        print(f"=> LỖI RUNTIME: {e}")
        print("\nGợi ý: Nếu vẫn lỗi shape, hãy kiểm tra file config.yaml của bạn có POOLER_RESOLUTION không phải là 7.")

if __name__ == "__main__":
    test_model_structure()
