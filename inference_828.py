import os
import numpy as np
import open_clip
import torch
from model import ResNetBERTModel
from inference_data_preprocessing import load_inference_data
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# --------------------------
# Configuration Parameters
# --------------------------
config = {
    'train_csv': None,
    'val_csv': None,
    'test_csv': 'D:/Codes/LineArtPred/data/csv/test4_有先验有跑图方便快速对比.csv',
    'image_dir': 'D:/Codes/JigsawPrediction/data/test4_images',
    'batch_size': 32,
    'checkpoints': [
        'models/coca_vit_b_32/checkpoints/best_model_epoch_3.pth',
        'models/coca_vit_b_32/checkpoints1/best_model_epoch_1.pth',
        'models/coca_vit_b_32/checkpoints2/best_model_epoch_1.pth'
    ],
    'use_gpu': torch.cuda.is_available(),
    'num_classes': 5,
    'use_features': {
        'use_full_image': True,
        'use_numeric_features': True,
        'use_text_hierarchy': True
    }
}

# --------------------------
# Model Initialization
# --------------------------
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
models = [load_model(ckpt, device, num_numeric_features, config['num_classes']) for ckpt in config['checkpoints']]

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

            # Predictions from model 2
            output = models[2](full_image, numeric_features, text_features)
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
output_csv_path = os.path.join(os.path.dirname(config['checkpoints'][0]), "test_predictions.csv")
test_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
