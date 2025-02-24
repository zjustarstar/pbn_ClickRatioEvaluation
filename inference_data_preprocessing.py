import os
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import random

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

        # Clean numeric features
        numeric_columns = ['是否blend', '是否修改']
        self.data[numeric_columns] = self.data[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

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

    def augment_image(self, image):
        """
        Applies augmentation to the given image with a combination of transformations.
        """
        if random.random() > 0.5:
            image = ImageOps.mirror(image)  # Horizontal flip
        if random.random() > 0.5:
            image = ImageOps.flip(image)  # Vertical flip
        if random.random() > 0.5:
            angle = random.randint(-30, 30)  # Random rotation
            image = image.rotate(angle)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))  # Randomly adjust brightness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))  # Randomly adjust contrast
        if random.random() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))  # Apply Gaussian blur
        if random.random() > 0.5:
            np_image = np.array(image)
            noise = np.random.normal(0, 10, np_image.shape).astype(np.uint8)
            np_image = np.clip(np_image + noise, 0, 255)
            image = Image.fromarray(np_image)

        return image

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
        if self.use_features['use_numeric_features']:
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

        # Ensure all features are valid
        if full_image is None or numeric_features is None or text_features is None:
            return None  # Skip invalid sample

        return full_image, numeric_features, text_features


def get_class_weights(dataset):
    """Compute class weights for handling imbalance."""
    labels = [
        dataset.get_label_from_intervals(
            row['点击率（双端合计）'],
            row['是否修改'],
            row.get('测图年月')  # 获取测图年月
        )
        for _, row in dataset.data.iterrows()
    ]
    class_counts = np.bincount(labels, minlength=4)
    weights = 1.0 / (class_counts + 1e-8)  # Avoid division by zero
    sample_weights = np.array([weights[label] for label in labels])
    return torch.DoubleTensor(sample_weights)


def get_class_counts(dataset):
    """统计每个类别的样本数量。"""
    labels = [
        dataset.get_label_from_intervals(
            row['点击率（双端合计）'],
            row['是否修改'],
            row.get('测图年月')  # 获取测图年月
        )
        for _, row in dataset.data.iterrows()
    ]
    class_counts = np.bincount(labels, minlength=4)
    return class_counts


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
    return inference_loader
