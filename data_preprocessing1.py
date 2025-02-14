import os
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import random


class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, preprocess, clip_tokenizer, use_features=None, augment=False,
                 target_classes=None):
        # Load and clean the dataset
        print(f"Loading dataset from: {csv_file}")
        self.data = pd.read_csv(csv_file)

        # # Remove duplicates based on 'picture_id'
        # duplicates = self.data[self.data['picture_id'].duplicated()]
        # if not duplicates.empty:
        #     print(f"Found {len(duplicates)} duplicate entries. Removing duplicates...")
        #     self.data = self.data.drop_duplicates(subset=['picture_id'], keep='first')

        # Clean numeric features
        numeric_columns = ['是否blend', '是否修改']
        self.data[numeric_columns] = self.data[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Initialize other parameters
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.clip_tokenizer = clip_tokenizer
        self.augment = augment
        self.target_classes = target_classes if target_classes is not None else [0, 3]

        # Default feature usage
        if use_features is None:
            use_features = {
                'use_full_image': True,
                'use_numeric_features': True,
                'use_text_hierarchy': True
            }
        self.use_features = use_features

        # Add calculated category column
        print("Calculating categories based on 点击率（双端合计） and 是否修改...")
        self.data['calculated_category'] = self.data.apply(
            lambda row: self.get_label_from_intervals(row['点击率（双端合计）'], row['是否修改']), axis=1
        )

        # Debug: Check for invalid rows
        invalid_rows = self.data[self.data['calculated_category'].isnull()]
        if not invalid_rows.empty:
            print(f"Found {len(invalid_rows)} rows with invalid category assignment:")
            print(invalid_rows[['点击率（双端合计）', '是否修改']])

        # Debug: Check overall distribution
        print("Dataset class distribution based on calculated categories:")
        class_distribution = self.data['calculated_category'].value_counts().sort_index()
        for category, count in class_distribution.items():
            print(f"Category {category}: {count} samples")

        # Debug: Verify 是否修改 field distribution
        print("Distribution of 是否修改:")
        print(self.data['是否修改'].value_counts())

        # Debug: Verify 点击率（双端合计） range
        print("Range of 点击率（双端合计）:")
        print(f"Min: {self.data['点击率（双端合计）'].min()}, Max: {self.data['点击率（双端合计）'].max()}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def balance_classes(self):
        """
        Balance the dataset by dynamically generating Cut-Mix samples for underrepresented classes (0 and 3).
        """
        labels = [self.get_label_from_intervals(row['点击率（双端合计）'], row['是否修改']) for _, row in self.data.iterrows()]
        class_counts = np.bincount(labels, minlength=4)
        max_count = max(class_counts)

        # Find underrepresented classes
        for class_idx in [0, 3]:
            class_indices = np.where(np.array(labels) == class_idx)[0]
            additional_samples = max_count - class_counts[class_idx]

            if additional_samples > 0:
                # Generate Cut-Mix samples for the class
                for _ in range(additional_samples):
                    idx1, idx2 = np.random.choice(class_indices, 2, replace=True)
                    new_sample = self.generate_cut_mix_sample(idx1, idx2)
                    self.data = pd.concat([self.data, pd.DataFrame([new_sample])], ignore_index=True)

        # 统计平衡后的类别数量
        labels_after_balance = [
            self.get_label_from_intervals(row['点击率（双端合计）'], row['是否修改']) for _, row in self.data.iterrows()
        ]
        class_counts_after_balance = np.bincount(labels_after_balance, minlength=4)
        print(f"Balanced class counts: {class_counts_after_balance}")

    def generate_cut_mix_sample(self, idx1, idx2):
        """
        Generate a new Cut-Mix sample by mixing two images and their metadata.
        """
        row1, row2 = self.data.iloc[idx1], self.data.iloc[idx2]

        # Load images
        image1_path = os.path.join(self.image_dir, row1['picture_url'])
        image2_path = os.path.join(self.image_dir, row2['picture_url'])
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')
        image2 = image2.resize(image1.size)

        # Generate Cut-Mix bounding box
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image1.size, lam)

        # Create new image
        np_image1 = np.array(image1)
        np_image2 = np.array(image2)
        np_image1[bby1:bby2, bbx1:bbx2, :] = np_image2[bby1:bby2, bbx1:bbx2, :]
        cut_mix_image = Image.fromarray(np_image1)

        # Save the new image temporarily
        cut_mix_path = f"cut_mix_{row1['picture_id']}_{row2['picture_id']}.jpg"
        cut_mix_image.save(os.path.join(self.image_dir, cut_mix_path))

        # Generate new metadata (use metadata from the first image)
        new_sample = row1.copy()
        new_sample['picture_url'] = cut_mix_path
        return new_sample

    def rand_bbox(self, size, lam):
        """
        Generate a random bounding box for Cut-Mix.
        """
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample the center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Calculate the bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def get_label_from_intervals(self, value, is_modified):
        """
        Determines the interval label for a given value of 点击率（双端合计） based on 是否修改.
        """
        if pd.isnull(value) or pd.isnull(is_modified):
            raise ValueError(f"Missing value detected: 点击率（双端合计）={value}, 是否修改={is_modified}")

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
                # print(f"Value: {value}, IsModified: {is_modified}, AssignedCategory: {idx}")
                return idx

        raise ValueError(f"Value {value} did not fit into any interval.")

    def augment_image(self, image):
        """
        针对线框图优化的数据增强方法。
        """
        if random.random() > 0.5:
            image = ImageOps.mirror(image)  # 水平翻转
        # if random.random() > 0.5:
        #     angle = random.uniform(-5, 5)  # 小幅旋转
        #     image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))  # 亮度调整范围
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.95, 1.05))  # 对比度调整范围
        # 禁用高斯模糊和噪声添加
        return image

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load and preprocess the full image
        image_path = os.path.join(self.image_dir, row['picture_url'])
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Determine the label for this sample based on 是否修改
        label = self.get_label_from_intervals(row['点击率（双端合计）'], row['是否修改'])

        # Apply data augmentation only during training for target classes
        if self.augment and label in self.target_classes:
            image = self.augment_image(image)

        full_image = self.preprocess(image) if self.use_features['use_full_image'] else None

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
            text_features = self.clip_tokenizer([text_input]).squeeze(0)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return full_image, numeric_features, text_features, label


def get_class_weights(dataset):
    """Compute class weights for handling imbalance."""
    labels = [dataset.get_label_from_intervals(row['点击率（双端合计）'], row['是否修改']) for _, row in dataset.data.iterrows()]
    class_counts = np.bincount(labels, minlength=4)
    weights = 1.0 / (class_counts + 1e-8)  # Avoid division by zero
    sample_weights = np.array([weights[label] for label in labels])
    return torch.DoubleTensor(sample_weights)


def get_class_counts(dataset):
    """统计每个类别的样本数量。"""
    labels = [dataset.get_label_from_intervals(row['点击率（双端合计）'], row['是否修改']) for _, row in dataset.data.iterrows()]
    class_counts = np.bincount(labels, minlength=4)
    return class_counts


def apply_undersampling(dataset):
    """
    对数据集执行欠采样，确保每个类别的数量相同。
    直接修改数据集的 DataFrame。
    """
    labels = [dataset.get_label_from_intervals(row['点击率（双端合计）']) for _, row in dataset.data.iterrows()]
    class_counts = get_class_counts(dataset)
    min_class_count = min(class_counts)  # 找到最小类别数量

    sampled_indices = []
    for class_idx in range(len(dataset.click_rate_intervals)):
        class_indices = np.where(np.array(labels) == class_idx)[0]
        sampled_class_indices = np.random.choice(class_indices, min_class_count, replace=False)
        sampled_indices.extend(sampled_class_indices)

    np.random.shuffle(sampled_indices)

    # 修改数据集，保留欠采样后的数据
    dataset.data = dataset.data.iloc[sampled_indices].reset_index(drop=True)
    return dataset


def load_data(train_csv, val_csv, test_csv, image_dir, preprocess, clip_tokenizer, batch_size, use_features):
    """
    Load datasets and return training, validation, and testing data loaders.
    Uses WeightedRandomSampler for handling class imbalance in the training set.
    """
    data_loaders = {}

    # Load training set
    if train_csv is not None:
        train_dataset = CustomDataset(
            csv_file=train_csv,
            image_dir=image_dir,
            preprocess=preprocess,
            clip_tokenizer=clip_tokenizer,
            use_features=use_features,
            augment=True,  # Enable data augmentation
            target_classes=[0, 3]  # Augmentation applies to classes 0 and 4
        )

        # Apply data balancing for training
        # train_dataset.balance_classes()
        # Compute class weights and apply WeightedRandomSampler
        sample_weights = get_class_weights(train_dataset)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # Create DataLoader
        data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        # Print class distribution for reference
        class_counts = get_class_counts(train_dataset)
        print(f"Training set class distribution: {class_counts}")
    else:
        print("Training data not provided. Skipping training set loading.")

    # Load validation set
    if val_csv is not None:
        val_dataset = CustomDataset(
            csv_file=val_csv,
            image_dir=image_dir,
            preprocess=preprocess,
            clip_tokenizer=clip_tokenizer,
            use_features=use_features,
            augment=False  # No augmentation for validation
        )
        data_loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        print("Validation data not provided. Skipping validation set loading.")

    # Load testing set
    if test_csv is not None:
        test_dataset = CustomDataset(
            csv_file=test_csv,
            image_dir=image_dir,
            preprocess=preprocess,
            clip_tokenizer=clip_tokenizer,
            use_features=use_features,
            augment=False  # No augmentation for testing
        )
        data_loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        print("Test data not provided. Skipping test set loading.")

    return data_loaders
