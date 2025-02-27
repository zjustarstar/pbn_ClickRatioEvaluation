import os
import shutil

import numpy as np
import open_clip
import torch
import yaml


# --------------------------
# Load Configuration with Validation
# --------------------------
def load_config(config_path='./config/config_inference.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    print("[Debug] 成功加载配置文件:", config)

    # 默认配置
    default_config = {
        "use_image_only": True,
        "incremental_training": False,
        "checkpoints": [],
        "image_only_checkpoints": [],
        "incremental_checkpoints": [],
        # "data_source": "csv",
        "test_csv": None,
        "image_dir": None,
        "zip_path": None,
        "batch_size": 32,
        "use_features": {
            "use_full_image": True,
            "use_numeric_features": False,
            "use_text_hierarchy": False
        },
        "use_gpu": False,
        "num_classes": 4
    }

    # **填充缺失的顶级字段**
    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    # **确保 `data_source` 存在**
    if "data_source" not in config:
        error_msg = """❌ `data_source` 缺失！
    ✅ 请检查 `config.yaml`，并手动设置 `data_source`，可选值: "csv" 或 "zip"。
    示例:
    data_source: "csv"
    """
        print(error_msg)
        raise KeyError(error_msg)

    # **确保 `data_source` 值合法**
    if config["data_source"] not in ["csv", "zip"]:
        error_msg = f"""❌ `data_source` 只能是 "csv" 或 "zip"，但当前值为 `{config["data_source"]}`
    ✅ 请检查 `config.yaml`，并修改 `data_source`，可选值:
    data_source: "csv"  # 从 CSV 读取
    data_source: "zip"  # 从 ZIP 读取
    """
        print(error_msg)
        raise ValueError(error_msg)

    print(f"[Debug] `data_source`: {config['data_source']}")  # ✅ 确保 `data_source` 结构 OK

    # **确保 `use_image_only` 是布尔值**
    if not isinstance(config["use_image_only"], bool):
        error_msg = f"""❌ `use_image_only` 需要是布尔值 True/False，但当前值为 `{config["use_image_only"]}` (类型: {type(config["use_image_only"])})
    ✅ 请检查 `config.yaml`，确保 `use_image_only` 仅为 `True` 或 `False`。
    示例:
    use_image_only: False
    """
        print(error_msg)
        raise TypeError(error_msg)

    print(f"[Debug] `use_image_only`: {config['use_image_only']}")

    # **检查 `checkpoints` 相关字段是否正确**
    if not isinstance(config["checkpoints"], list) or not config["checkpoints"]:
        error_msg = """❌ `checkpoints` 需要是一个非空列表。
✅ 请检查 `config.yaml`，示例:
checkpoints:
  - path/to/your_model.pth
"""
        print(error_msg)
        raise ValueError(error_msg)

    # **确保 `use_image_only` 和 `incremental_training` 是布尔值**
    if not isinstance(config["use_image_only"], bool):
        error_msg = f"""❌ `use_image_only` 需要是布尔值 True/False，但当前值为 `{config["use_image_only"]}` (类型: {type(config["use_image_only"])})
✅ 请检查 `config.yaml`，确保 `use_image_only` 仅为 `True` 或 `False`。
示例:
use_image_only: False
"""
        print(error_msg)
        raise TypeError(error_msg)

    if not isinstance(config["incremental_training"], bool):
        error_msg = f"""❌ `incremental_training` 需要是布尔值 True/False，但当前值为 `{config["incremental_training"]}` (类型: {type(config["incremental_training"])})
✅ 请检查 `config.yaml`，确保 `incremental_training` 仅为 `True` 或 `False`。
示例:
incremental_training: True
"""
        print(error_msg)
        raise TypeError(error_msg)

    print("[Debug] `use_features` 结构正确:", config["use_features"])  # ✅ 确保 `use_features` 结构 OK
    return config


# 读取配置
try:
    config = load_config()
except Exception as e:
    print(f"[Error] 配置加载失败: {e}")
    exit(1)  # 终止程序


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

# **[Debug] 打印 `checkpoints`**
print(f"[Debug] 加载的 `checkpoints`: {checkpoints}")

# **确保 `checkpoints` 不是空列表**
if not checkpoints:
    error_msg = f"""❌ `checkpoints` 不能为空！
✅ 请检查 `config.yaml`，如果 `use_image_only: True`，必须填写 `image_only_checkpoints`:
示例:
image_only_checkpoints:
  - path/to/image_model.pth
"""
    print(error_msg)
    raise ValueError(error_msg)

# **逐个检查 `checkpoints` 是否存在**
missing_checkpoints = [ckpt for ckpt in checkpoints if not os.path.isfile(ckpt)]
if missing_checkpoints:
    error_msg = f"""❌ 发现以下 `checkpoint` 文件不存在:
{missing_checkpoints}

✅ 请检查 `config.yaml` 中的路径，确保文件存在。
示例:
image_only_checkpoints:
  - models/image_only_model/checkpoints5/best_model.pth
"""
    print(error_msg)
    raise FileNotFoundError(error_msg)

print("[Debug] 所有 `checkpoints` 文件存在，继续执行模型加载...")

if config['data_source'] not in ['csv', 'zip']:
    raise ValueError("配置文件 `data_source` 仅支持 'csv' 或 'zip'.")

if config['data_source'] == 'csv' and not config['test_csv']:
    raise ValueError("`data_source` 设为 'csv' 时，`test_csv` 不能为空.")
if config['data_source'] == 'zip' and not config['zip_path']:
    raise ValueError("`data_source` 设为 'zip' 时，`zip_path` 不能为空.")

# 确定使用的模型
if incremental_training:
    from partial_finetune_model import ResNetBERTModel  # 增量训练模型
else:
    from base_model import ResNetBERTModel  # 基础模型（通用）

from inference_data_preprocessing import load_inference_data, extract_zip_and_load_data
from torchvision import transforms
import pandas as pd

def load_model(checkpoint_path, device, num_numeric_features, num_classes):
    """Initializes and loads a model from a checkpoint with error handling."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 文件未找到: {checkpoint_path}")

    try:
        model = ResNetBERTModel(num_numeric_features=num_numeric_features, num_classes=num_classes)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model successfully loaded from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"[Error] 加载模型失败: {e}")
        exit(1)  # 终止程序


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

