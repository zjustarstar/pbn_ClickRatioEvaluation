import os
import shutil

import numpy as np
import open_clip
import torch
import yaml


# --------------------------
# Load Configuration
# --------------------------
def load_config(config_path='./config/config_inference.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_config()

# --------------------------
# Model Initialization
# --------------------------
use_image_only = config.get('use_image_only', False)
incremental_training = config.get('incremental_training', False)

# 确定使用的 checkpoint
if use_image_only:
    checkpoints = config['image_only_checkpoints']
elif incremental_training:
    checkpoints = config['incremental_checkpoints']
else:
    checkpoints = config['checkpoints']

# 确定使用的模型
if incremental_training:
    from partial_finetune_model import ResNetBERTModel  # 增量训练模型
else:
    from base_model import ResNetBERTModel  # 基础模型（通用）

from inference_data_preprocessing import load_inference_data, extract_zip_and_load_data
from torchvision import transforms
import pandas as pd

def load_model(checkpoint_path, device, num_numeric_features, num_classes):
    """Initializes and loads a model from a checkpoint."""
    model = ResNetBERTModel(num_numeric_features=num_numeric_features, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

# Image preprocessing
image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------
# Data Loading
# --------------------------
if config['data_source'] == 'zip':
    test_loader, temp_dir = extract_zip_and_load_data(
        zip_path=config['zip_path'],
        preprocess=image_preprocess,
        clip_tokenizer=open_clip.tokenize,
        batch_size=config['batch_size'],
        use_features=config['use_features']
    )
else:
    test_loader, temp_dir = load_inference_data(
        csv_file=config['test_csv'],
        image_dir=config['image_dir'],
        preprocess=image_preprocess,
        clip_tokenizer=open_clip.tokenize,
        batch_size=config['batch_size'],
        use_features=config['use_features']
    ), None  # CSV 加载方式没有 temp_dir

# Set device (GPU or CPU)
device = torch.device("cuda" if config['use_gpu'] else "cpu")

# Load all models
num_numeric_features = 2  # "是否blend" and "色块数"
models = [load_model(ckpt, device, num_numeric_features, config['num_classes']) for ckpt in checkpoints]

# --------------------------
# Inference Function
# --------------------------
def inference(models, data_loader, device):
    """Performs inference with multiple models and uses only preds[2]."""
    all_final_preds = []
    all_final_confs = []

    with torch.no_grad():
        for batch in data_loader:
            # Unpack batch
            if len(batch) == 3:
                full_image, numeric_features, text_features = batch
            else:
                raise ValueError("Unexpected number of items in the batch. Expected 3 (full_image, numeric_features, text_features).")

            full_image = full_image.to(device) if full_image is not None else None
            numeric_features = numeric_features.to(device) if numeric_features is not None else None
            text_features = text_features.to(device) if text_features is not None else None

            # Predictions from model 0
            output = models[0](full_image, numeric_features, text_features)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)

            all_final_preds.extend(pred.cpu().numpy())
            all_final_confs.extend(prob[range(len(pred)), pred].cpu().numpy())

    return np.array(all_final_preds), np.array(all_final_confs)

# --------------------------
# Execute Inference
# --------------------------

# Perform inference
final_predictions, final_confidences = inference(models, test_loader, device)

# **读取对应的 CSV**
if config['data_source'] == 'zip':
    test_csv_path = os.path.join(temp_dir, "extracted_test.csv")  # 读取 ZIP 生成的 CSV
else:
    test_csv_path = config['test_csv']  # 读取原始 CSV

test_df = pd.read_csv(test_csv_path)  # **正确加载与推理数据一致的 CSV**

# 确保 test_df 和 预测结果 长度匹配
if len(test_df) != len(final_predictions):
    print(f"[Warning] Mismatch: Predictions {len(final_predictions)}, DataFrame {len(test_df)}")
    test_df = test_df.iloc[:len(final_predictions)]  # **修正 test_df 长度**

# 保存预测结果
test_df['final_predicted_label'] = final_predictions
test_df['final_confidence'] = final_confidences

output_csv_path = os.path.join(os.path.dirname(checkpoints[0]), "test_predictions.csv")
test_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")

# --------------------------
# Cleanup Temporary Directory (if used)
# --------------------------
if temp_dir:
    shutil.rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} deleted.")

