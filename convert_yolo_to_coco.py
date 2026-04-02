import os
import json
from PIL import Image

def yolo_to_coco_with_hierarchical_labels(
    yolo_images_dir,
    yolo_labels_dir,
    output_json_path,
    image_ext=".jpg"
):
    """
    Converts a single-class YOLO dataset to a Detectron2/COCO-compatible JSON format.
    Injects placeholder values for 'patho' and 'jaw' classes, required by the customized DiffusionDet.
    """
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "tooth_0"}, # Need to properly define classes if available
            {"id": 2, "name": "tooth_1"}
        ]
    }
    
    annotation_id = 1
    
    for image_name in os.listdir(yolo_images_dir):
        if not image_name.endswith(image_ext):
            continue
            
        image_path = os.path.join(yolo_images_dir, image_name)
        label_name = image_name.replace(image_ext, ".txt")
        label_path = os.path.join(yolo_labels_dir, label_name)
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue
            
        image_id = int(''.join(filter(str.isdigit, image_name)) or 0)
        
        coco_format["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_path, # usually relative, using absolute for now or can configure
        })
        
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id, x_center, y_center, w, h = map(float, parts[:5])
                
                # YOLO format is normalized. Convert to absolute pixel values
                abs_x_center = x_center * width
                abs_y_center = y_center * height
                abs_w = w * width
                abs_h = h * height
                
                # COCO bbox format: [x_min, y_min, width, height]
                x_min = abs_x_center - (abs_w / 2)
                y_min = abs_y_center - (abs_h / 2)
                
                # We need to map the single class YOLO to the hierarchical format
                # For placeholder:
                # category_id = tooth number (e.g., 0 to 31)
                # patho = pathology status (e.g., 0 to 3)
                # jaw = jaw area (0 or 1)
                
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,         # Placeholder Tooth Number class (1-indexed usually in COCO)
                    "patho": 0,               # Placeholder Pathology class
                    "jaw": 0,                 # Placeholder Jaw class
                    "bbox": [x_min, y_min, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
                annotation_id += 1
                
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=4)
        
    print(f"Successfully converted.")

if __name__ == "__main__":
    train_images = r"C:\Users\My Computer\Downloads\Dự án TAD-AI-3\archive\Data_Training_caries\train\images"
    train_labels = r"C:\Users\My Computer\Downloads\Dự án TAD-AI-3\archive\Data_Training_caries\train\labels"
    out_json = r"C:\Users\My Computer\Downloads\Dự án TAD-AI-3\archive\Data_Training_caries\train_coco_hierarchical.json"
    
    yolo_to_coco_with_hierarchical_labels(train_images, train_labels, out_json)
