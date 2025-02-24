import os

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
if config.get('incremental_training', False):
    from partial_finetune_model import ResNetBERTModel
    checkpoints = config['incremental_checkpoints']
else:
    from base_model import ResNetBERTModel
    checkpoints = config['checkpoints']

from inference_data_preprocessing import load_inference_data
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

# Load test data
test_loader = load_inference_data(
    config['test_csv'],
    config['image_dir'], image_preprocess, open_clip.tokenize, config['batch_size'],
    config['use_features']
)

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

# Save predictions to CSV
test_df = pd.read_csv(config['test_csv'])
test_df['final_predicted_label'] = final_predictions
test_df['final_confidence'] = final_confidences
output_csv_path = os.path.join(os.path.dirname(checkpoints[0]), "test_predictions.csv")
test_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
