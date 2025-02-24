import os

import numpy as np
import open_clip
import torch
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau


# --------------------------
# Load Configuration
# --------------------------
def load_config(config_path='./config/config_train.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 确保数值参数是正确的类型
    config['learning_rate'] = float(config['learning_rate'])

    return config


config = load_config()
# 根据是否增量训练选择加载的模型
if config['incremental_training']:
    from partial_finetune_model import ResNetBERTModel
    train_params = config['incremental_params']
else:
    from base_model import ResNetBERTModel
    train_params = {
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "patience": config["patience"],
        "checkpoint_dir": config["checkpoint_dir"]
    }

from data_preprocessing import load_data as load_data

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms


# --------------------------
# Model Initialization
# --------------------------
num_numeric_features = 2  # "是否blend" and "色块数"
model_wrapper = ResNetBERTModel(num_numeric_features=num_numeric_features, num_classes=config['num_classes'])

# 如果是增量训练，加载预训练模型权重
if (config['incremental_training'] and os.path.exists

    (config['incremental_model_path'])):
    model_wrapper.load_state_dict(torch.load(config['incremental_model_path']))
    print(f"Loaded incremental model weights from {config['incremental_model_path']}")

resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
data_loaders = load_data(
    config['train_csv'], config['val_csv'], config['test_csv'],
    config['image_dir'], image_preprocess, open_clip.tokenize, train_params['batch_size'],
    config['use_features']
)

# Set device (GPU or CPU)
device = torch.device("cuda" if config['use_gpu'] else "cpu")
model_wrapper = model_wrapper.to(device)

# --------------------------
# Loss and Optimizer
# --------------------------
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
if config['incremental_training']:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_wrapper.parameters()),
                           lr=float(train_params['learning_rate']), weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
else:
    optimizer = optim.Adam(model_wrapper.parameters(), lr=train_params['learning_rate'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Ensure checkpoint directory exists
os.makedirs(train_params['checkpoint_dir'], exist_ok=True)

# 梯度裁剪函数
def clip_gradients(model, max_norm=1.0):
    """应用梯度裁剪"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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

        # 增量训练时可以选择加大梯度裁剪
        if config['incremental_training']:
            clip_gradients(model_wrapper, max_norm=1.0)

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

def save_best_model(model, epoch, checkpoint_dir, num_classes):
    """Saves the best model during training with num_classes info in filename."""
    best_model_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch + 1}_num_classes_{num_classes}_batch_size_{train_params['batch_size']}.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"Best model saved at {best_model_path}")

# --------------------------
# Training Loop
# --------------------------
best_val_loss = float('inf')
early_stop_counter = 0
patience_limit = train_params['patience']
last_model_path = os.path.join(train_params['checkpoint_dir'], "last_model.pth")  # 保存最后一个模型的路径

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
        save_best_model(model_wrapper, epoch, train_params['checkpoint_dir'], config['num_classes'])

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