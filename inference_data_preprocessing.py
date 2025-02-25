import os
import tempfile
import zipfile

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, preprocess, clip_tokenizer, use_features=None, augment=False,
                 target_classes=None, mode='train'):
        """
        Initialize the dataset.
        :param mode: 'train', 'val', or 'test'. Determines if target values are required.
        """
        # Load and clean the dataset
        print(f"Loading dataset from: {csv_file}")
        self.data = pd.read_csv(csv_file)

        # Initialize other parameters
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.clip_tokenizer = clip_tokenizer
        self.augment = augment
        self.target_classes = target_classes if target_classes is not None else [0, 3]
        self.mode = mode  # Mode can be 'train', 'val', or 'test'

        # Default feature usage
        if use_features is None:
            use_features = {
                'use_full_image': True,
                'use_numeric_features': True,
                'use_text_hierarchy': True
            }
        self.use_features = use_features

        if self.use_features['use_numeric_features']:
            numeric_columns = ['是否blend', '是否修改']
            self.data[numeric_columns] = self.data[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        else:
            self.data['是否blend'] = 0
            self.data['是否修改'] = 1

        # Add calculated category column for training and validation
        if self.mode != 'test':
            print("Calculating categories based on 点击率（双端合计）, 是否修改, and 测图年月...")
            self.data['calculated_category'] = self.data.apply(
                lambda row: self.get_label_from_intervals(
                    row['点击率（双端合计）'],
                    row['是否修改'],
                    row.get('测图年月')  # 获取测图年月
                ),
                axis=1
            )
            # Filter out rows where the label is None (i.e., invalid rows)
            self.data = self.data.dropna(subset=['calculated_category'])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def get_label_from_intervals(self, value, is_modified, measure_month=None):
        """
        Determines the interval label for a given value of 点击率（双端合计） based on 是否修改 and 测图年月.
        """
        if pd.isnull(value) or pd.isnull(is_modified):
            raise ValueError(f"Missing value detected: 点击率（双端合计）={value}, 是否修改={is_modified}")

        if measure_month == "2024年7月":
            # 特殊分类规则
            intervals = [
                (float('-inf'), 0.09),  # Class 0
                (0.09, 0.12),  # Class 1
                (0.15, 0.18),  # Class 2
                (0.18, float('inf'))  # Class 3
            ]
            # Check if the value falls within any of the defined intervals, else return None
            for idx, (low, high) in enumerate(intervals):
                if low <= value < high:
                    return idx
            return None  # Return None if the value doesn't match any interval
        else:
            # 默认分类规则
            if is_modified == 0:
                intervals = [
                    (float('-inf'), 0.12),  # Class 0
                    (0.12, 0.15),  # Class 1
                    (0.15, 0.18),  # Class 2
                    (0.18, float('inf'))  # Class 3
                ]
            elif is_modified == 1:
                intervals = [
                    (float('-inf'), 0.09),  # Class 0
                    (0.09, 0.12),  # Class 1
                    (0.12, 0.15),  # Class 2
                    (0.15, float('inf'))  # Class 3
                ]
            else:
                raise ValueError(f"Invalid '是否修改' value: {is_modified}")

        for idx, (low, high) in enumerate(intervals):
            if low <= value < high:
                return idx

        raise ValueError(f"Value {value} did not fit into any interval.")


    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load and preprocess the full image
        image_path = os.path.join(self.image_dir, row['picture_url'])
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            full_image = self.preprocess(image) if self.use_features['use_full_image'] else None
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            full_image = None

        # Load numerical features
        numeric_features = []
        numeric_features.extend([row['是否blend'], row['是否修改']])
        numeric_features = torch.tensor(numeric_features, dtype=torch.float32) if numeric_features else None

        # Tokenize text features
        text_features = None
        if self.use_features['use_text_hierarchy']:
            text_input = " ".join([
                row['一级主要元素Tag'], row['二级主要元素Tag'], row['三级主要元素Tag'],
                row['一级环境Tag'], row['二级环境Tag'], row['三级环境Tag'],
                row['一级次要元素Tag'], row['二级次要元素Tag'], row['三级次要元素Tag'],
                row['图片类型']
            ])
            try:
                text_features = self.clip_tokenizer([text_input]).squeeze(0)
            except Exception as e:
                print(f"Error tokenizing text: {e}")
        else:
            text_features = torch.zeros(512)

        # Ensure all features are valid
        if full_image is None or numeric_features is None or text_features is None:
            return None  # Skip invalid sample

        return full_image, numeric_features, text_features


def load_inference_data(csv_file, image_dir, preprocess, clip_tokenizer, batch_size, use_features):
    """
    Load dataset for inference and return a DataLoader.
    """
    inference_dataset = CustomDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        preprocess=preprocess,
        clip_tokenizer=clip_tokenizer,
        use_features=use_features,
        augment=False,
        mode='test'
    )

    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    return inference_loader, None


def extract_zip_and_load_data(zip_path, preprocess, clip_tokenizer, batch_size, use_features):
    """
    解压 ZIP 文件并加载数据到 DataLoader，并新增 picture_id 列（去掉后缀的图片名）。
    """
    # 创建临时目录解压文件
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # 设定解压后的图片目录
    extracted_image_dir = temp_dir

    # 查找所有图片文件
    image_files = [f for f in os.listdir(extracted_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 生成 `picture_id` 列（去掉后缀的文件名）
    picture_ids = [os.path.splitext(f)[0] for f in image_files]

    # 生成 CSV 以便后续加载
    extracted_csv_path = os.path.join(temp_dir, "extracted_test.csv")
    image_df = pd.DataFrame({'picture_url': image_files, 'picture_id': picture_ids})  # 增加 picture_id
    image_df.to_csv(extracted_csv_path, index=False)

    # 读取数据
    inference_dataset = CustomDataset(
        csv_file=extracted_csv_path,
        image_dir=extracted_image_dir,
        preprocess=preprocess,
        clip_tokenizer=clip_tokenizer,
        use_features=use_features,
        augment=False,
        mode='test'
    )

    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    return inference_loader, temp_dir