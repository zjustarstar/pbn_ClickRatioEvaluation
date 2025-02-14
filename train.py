import os
import numpy as np
import open_clip
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import ResNetBERTModel
from data_preprocessing import load_data as load_data
from data_preprocessing1 import load_data as load_data1
from data_preprocessing2 import load_data as load_data2

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

# --------------------------
# Configuration Parameters
# --------------------------
config = {
    'train_csv': 'D:/Codes/LineArtPred/data/csv/balanced_data4/train_dataset_balanced.csv',
    'val_csv': 'D:/Codes/LineArtPred/data/csv/balanced_data4/val_dataset.csv',
    'test_csv': 'D:/Codes/LineArtPred/data/csv/balanced_data4/test_dataset.csv',
    'image_dir': 'D:/Codes/JigsawPrediction/data/images',
    'batch_size': 32,
    'learning_rate': 5e-5,
    'num_epochs': 100,
    'patience': 3,
    'checkpoint_dir': 'models/coca_vit_b_32/checkpoints5/',
    'use_gpu': torch.cuda.is_available(),
    'num_classes': 4,
    'use_features': {
        'use_full_image': True,
        'use_numeric_features': True,
        'use_text_hierarchy': True
    }
}

# --------------------------
# Model Initialization
# --------------------------
num_numeric_features = 2  # "是否blend" and "色块数"
model_wrapper = ResNetBERTModel(num_numeric_features=num_numeric_features, num_classes=config['num_classes'])
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
data_loaders = load_data2(
    config['train_csv'], config['val_csv'], config['test_csv'],
    config['image_dir'], image_preprocess, open_clip.tokenize, config['batch_size'],
    config['use_features']
)

# Set device (GPU or CPU)
device = torch.device("cuda" if config['use_gpu'] else "cpu")
model_wrapper = model_wrapper.to(device)

# --------------------------
# Loss and Optimizer
# --------------------------
class CBLoss(nn.Module):
    """Class-Balanced Loss to handle class imbalance."""
    def __init__(self, beta, num_classes):
        super(CBLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets):
        effective_num = 1.0 - torch.pow(self.beta, targets.bincount(minlength=self.num_classes))
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * self.num_classes
        weights = weights.to(logits.device)

        return F.cross_entropy(logits, targets, weight=weights)

# criterion = CBLoss(beta=0.99, num_classes=config['num_classes'])
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.25, gamma=2.0)  # 使用 Focal Loss
optimizer = optim.Adam(model_wrapper.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Ensure checkpoint directory exists
os.makedirs(config['checkpoint_dir'], exist_ok=True)

# --------------------------
# Training and Validation Functions
# --------------------------
def train_one_epoch(epoch, model, data_loader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0

    for i, (full_image, numeric_features, text_features, labels) in enumerate(data_loader):
        full_image = full_image.to(device) if full_image is not None else None
        numeric_features = numeric_features.to(device) if numeric_features is not None else None
        text_features = text_features.to(device) if text_features is not None else None
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(full_image, numeric_features, text_features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print loss every 10 steps
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    """Validates the model on the validation dataset."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for full_image, numeric_features, text_features, labels in data_loader:
            full_image = full_image.to(device) if full_image is not None else None
            numeric_features = numeric_features.to(device) if numeric_features is not None else None
            text_features = text_features.to(device) if text_features is not None else None
            labels = labels.to(device)

            outputs = model(full_image, numeric_features, text_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    return total_loss / len(data_loader), accuracy

# --------------------------
# Early Stopping Logic
# --------------------------
def early_stopping_criteria(val_loss, best_val_loss, patience_counter, patience_limit):
    """Determines whether to stop training based on validation loss."""
    stop_training = False
    update_best_model = False

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        update_best_model = True
    else:
        patience_counter += 1

    if patience_counter >= patience_limit:
        stop_training = True

    return stop_training, patience_counter, update_best_model, best_val_loss

def save_best_model(model, epoch, checkpoint_dir):
    """Saves the best model during training."""
    best_model_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"Best model saved at {best_model_path}")

# --------------------------
# Training Loop
# --------------------------
best_val_loss = float('inf')
early_stop_counter = 0
patience_limit = config['patience']
last_model_path = os.path.join(config['checkpoint_dir'], "last_model.pth")  # 保存最后一个模型的路径

for epoch in range(config['num_epochs']):
    train_loss = train_one_epoch(epoch, model_wrapper, data_loaders['train'], optimizer, criterion, device)
    val_loss, val_acc = validate(model_wrapper, data_loaders['val'], criterion, device)

    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], "
          f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    scheduler.step(val_loss)

    stop_training, early_stop_counter, update_best_model, best_val_loss = early_stopping_criteria(
        val_loss, best_val_loss, early_stop_counter, patience_limit
    )

    if update_best_model:
        save_best_model(model_wrapper, epoch, config['checkpoint_dir'])

    # Save the last model after every epoch
    torch.save(model_wrapper.state_dict(), last_model_path)

    if stop_training:
        print(f"Early stopping triggered at epoch {epoch + 1}.")
        break

# --------------------------
# Test Set Evaluation
# --------------------------
def evaluate_on_test_set(model, data_loader, device):
    """Evaluates the model on the test dataset with comprehensive metrics."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for full_image, numeric_features, text_features, labels in data_loader:
            full_image = full_image.to(device) if full_image is not None else None
            numeric_features = numeric_features.to(device) if numeric_features is not None else None
            text_features = text_features.to(device) if text_features is not None else None
            labels = labels.to(device)

            outputs = model(full_image, numeric_features, text_features)
            _, preds = torch.max(outputs, 1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, digits=4)

    print(f"Test set evaluation results:\nAccuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    return accuracy, conf_matrix, class_report

# --------------------------
# Test Set Evaluation
# --------------------------
# Load the last model for evaluation
model_wrapper.load_state_dict(torch.load(last_model_path))  # 加载最后一个模型
model_wrapper = model_wrapper.to(device)

accuracy, conf_matrix, class_report = evaluate_on_test_set(model_wrapper, data_loaders['test'], device)
print(f"Final test set results (Last Model):\nAccuracy = {accuracy:.4f}")