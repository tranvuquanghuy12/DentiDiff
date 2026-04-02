import json
import os

def fix_coco_json(json_path, output_path):
    print(f"Reading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    print(f"Categories: {len(data['categories'])}")
    
    fixed_images = 0
    for img in data['images']:
        # Current path: C:\Users\My Computer\Downloads\Dự án TAD-AI-3\archive\Data_Training_caries\train\images\image_1.jpg
        # We want to keep only the filename or make it relative to the dataset root
        full_path = img['file_name']
        filename = os.path.basename(full_path)
        img['file_name'] = filename
        fixed_images += 1
    
    print(f"Fixed {fixed_images} image paths.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved fixed JSON to {output_path}")

if __name__ == "__main__":
    input_path = r"d:\Project AI\Dự án TAD-AI-3\archive\Data_Training_caries\train_coco_hierarchical.json"
    output_path = r"d:\Project AI\Dự án TAD-AI-3\archive\Data_Training_caries\train_coco_hierarchical_fixed.json"
    fix_coco_json(input_path, output_path)
