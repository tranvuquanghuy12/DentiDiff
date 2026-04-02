import torch
import os
import numpy as np
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from diffusiondet import add_diffusiondet_config

def evaluate_hierarchical(cfg_path, model_weights):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    dataset_name = cfg.DATASETS.TEST[0]
    
    data_loader = build_detection_test_loader(cfg, dataset_name)
    
    print(f"Starting evaluation on {dataset_name}...")
    
    all_gt_patho = []
    all_pred_patho = []
    all_scores = []
    
    # Custom evaluation for hierarchical labels
    with torch.no_grad():
        for inputs in tqdm(data_loader):
            outputs = predictor.model(inputs)
            for input_per_img, output_per_img in zip(inputs, outputs):
                instances = output_per_img["instances"].to("cpu")
                gt_instances = input_per_img["instances"].to("cpu")
                
                # To simplify: we look at matched boxes or just overall class distributions
                # For a full F1/AP on pathology, we usually need to match pred boxes to GT boxes
                # detectron2's COCOEvaluator handles the box matching for 'pred_classes'.
                # We can extend it or do a manual match here.
                
                if len(instances) == 0:
                    continue
                
                # Simple implementation: log top predictions
                if hasattr(instances, "pred_patho"):
                    all_pred_patho.extend(instances.pred_patho.tolist())
                    all_scores.extend(instances.scores.tolist())
                
                if hasattr(gt_instances, "gt_patho"):
                    all_gt_patho.extend(gt_instances.gt_patho.tolist())

    print("--- Hierarchical Evaluation Summary ---")
    # Note: Proper AP/F1 requires IoU matching which is complex to re-implement here.
    # We recommend using the COCOEvaluator for the main task and this for additional stats.
    
    # Run standard COCO evaluation
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/eval")
    results = inference_on_dataset(predictor.model, data_loader, evaluator)
    return results

if __name__ == "__main__":
    # Example usage:
    # python eval_results.py --config configs/denti_diffusion_cfg.yaml --weights output/model_final.pth
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="DiffusionDet/configs/denti_diffusion_cfg.yaml")
    parser.add_argument("--weights", default="output/model_final.pth")
    args = parser.parse_args()
    
    if os.path.exists(args.config) and os.path.exists(args.weights):
        evaluate_hierarchical(args.config, args.weights)
    else:
        print("Config or weights not found. Please check paths.")
