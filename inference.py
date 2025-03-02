import os
import shutil
from tqdm import tqdm
from config_loader import load_config_infer

import numpy as np
import open_clip
import torch


def load_model(checkpoint_path, config, device, num_numeric_features, num_classes):
    """Initializes and loads a model from a checkpoint with error handling."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 文件未找到: {checkpoint_path}")

    incremental_training = config.get('incremental_training', False)
    # 确定使用的模型
    if incremental_training:
        from partial_finetune_model import ResNetBERTModel  # 增量训练模型
    else:
        from base_model import ResNetBERTModel  # 基础模型（通用）

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


from inference_data_preprocessing import load_inference_data, extract_zip_and_load_data
from torchvision import transforms
import pandas as pd

def load_data(config):
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

    return test_loader, temp_dir


# --------------------------
# Inference Function
# --------------------------
def inference(models, data_loader, device):
    """Performs inference with multiple models and uses only preds[2]."""
    all_final_preds = []
    all_final_confs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing", unit="batch"):
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


def clear_output_folders(base_folder):
    """清理输出文件夹中的0、1、2、3分类文件夹，仅清空需要清理的文件夹。"""
    for label in ['0', '1', '2', '3']:
        folder_path = os.path.join(base_folder, label)
        if os.path.exists(folder_path):
            # 仅删除文件，不删除文件夹本身
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(folder_path)


def save_predictions_to_folders(test_df, output_base_folder, file_name):
    """根据预测结果将对应的图像文件复制到对应的文件夹中。"""
    output_folder = os.path.join(output_base_folder, os.path.splitext(os.path.basename(file_name))[0])
    os.makedirs(output_folder, exist_ok=True)
    clear_output_folders(output_folder)

    for label in [0, 1, 2, 3]:
        label_folder = os.path.join(output_folder, str(label))
        os.makedirs(label_folder, exist_ok=True)

    for _, row in test_df.iterrows():
        image_name = f"{row['picture_id']}.jpg"
        predicted_label = row['final_predicted_label']

        source_path = row['picture_path']
        target_folder = os.path.join(output_folder, str(predicted_label))
        target_path = os.path.join(target_folder, image_name)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)


# --------------------------
# Execute Inference
# --------------------------
def main():
    # 读取配置
    try:
        config = load_config_infer()
        # Set device (GPU or CPU)
        device = torch.device("cuda" if config['use_gpu'] else "cpu")

        use_image_only = config.get('use_image_only', False)
        incremental_training = config.get('incremental_training', False)
        # 确定使用的 checkpoint
        if use_image_only:
            checkpoints = config['image_only_checkpoints']
        elif incremental_training:
            checkpoints = config['incremental_checkpoints']
        else:
            checkpoints = config['checkpoints']
    except Exception as e:
        print(f"[Error] 配置加载失败: {e}")
        exit(1)  # 终止程序

    # Load all models
    test_loader, temp_dir = load_data(config)
    num_numeric_features = 2  # "是否blend" and "色块数"
    models = [load_model(ckpt, config, device, num_numeric_features, config['num_classes']) for ckpt in checkpoints]

    # Perform inference
    final_predictions, final_confidences = inference(models, test_loader, device)

    # **读取对应的 CSV**
    if config['data_source'] == 'zip':
        zip_name = os.path.splitext(os.path.basename(config['zip_path']))[0]  # 获取 zip 文件名（不含后缀）
        test_csv_path = os.path.join(temp_dir, f"{zip_name}.csv")  # 读取 ZIP 生成的 CSV
    else:
        test_csv_path = config['test_csv']  # 读取原始 CSV

    test_df = pd.read_csv(test_csv_path)  # **正确加载与推理数据一致的 CSV**

    # 确保 test_df 和 预测结果 长度匹配
    if len(test_df) != len(final_predictions):
        print(f"[Warning] Mismatch: Predictions {len(final_predictions)}, DataFrame {len(test_df)}")
        test_df = test_df.iloc[:len(final_predictions)]  # **修正 test_df 长度**

    # 保存预测结果
    test_df['final_predicted_label'] = final_predictions

    output_base_folder = os.path.dirname(test_csv_path) if config['data_source'] == 'csv' else os.path.dirname(
        config['zip_path'])
    image_dir = config['image_dir']
    save_predictions_to_folders(test_df, output_base_folder, os.path.basename(test_csv_path))
    print(f"Predictions saved into folders under {output_base_folder}")

    # --------------------------
    # Cleanup Temporary Directory (if used)
    # --------------------------
    if temp_dir:
        shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} deleted.")


if __name__ == "__main__":
    main()

