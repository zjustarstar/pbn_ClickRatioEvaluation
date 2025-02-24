import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer


class NumericTransformer(nn.Module):
    """
    Transformer module for processing numerical features.
    """
    def __init__(self, num_features, embed_dim, num_heads=8, num_layers=2):
        super(NumericTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch_size, num_features) -> (batch_size, seq_len=1, embed_dim)
        x = self.transformer_encoder(x)
        return x.squeeze(1)  # (batch_size, seq_len=1, embed_dim) -> (batch_size, embed_dim)


class ResNetBERTModel(nn.Module):
    def __init__(self, num_numeric_features, num_classes=5):
        super(ResNetBERTModel, self).__init__()

        # 1️⃣ 加载 ResNet，并冻结部分层
        resnet = models.resnet50(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉 FC 层
        resnet_feature_dim = resnet.fc.in_features

        # 冻结 ResNet 低层，仅调整 layer3、layer4 和 fc 层
        for param in resnet.conv1.parameters():
            param.requires_grad = False
        for param in resnet.layer1.parameters():
            param.requires_grad = False
        for param in resnet.layer2.parameters():
            param.requires_grad = False
        for param in resnet.layer3.parameters():
            param.requires_grad = True  # 允许微调
        for param in resnet.layer4.parameters():
            param.requires_grad = True  # 允许微调

        # 2️⃣ Numeric Transformer 部分，保持可训练
        embed_dim = 128
        self.numeric_transformer = NumericTransformer(num_numeric_features, embed_dim)
        numeric_feature_dim = embed_dim  # Transformer 输出维度

        # 3️⃣ 加载 BERT，并冻结部分层
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_feature_dim = 768  # BERT 输出的 [CLS] 维度

        # 只微调 BERT 的后两层
        for name, param in self.bert_model.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True  # 允许微调
            else:
                param.requires_grad = False  # 冻结其余层

        # 4️⃣ 合并所有特征
        combined_feature_dim = resnet_feature_dim + numeric_feature_dim + text_feature_dim

        # 5️⃣ 分类层（全连接层）
        self.fc1 = nn.Linear(combined_feature_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout 和激活函数
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()


    def forward(self, line_image, numeric_features, text_features):
        """
        Forward pass for the model.

        Args:
            line_image (torch.Tensor): Line drawing image tensor (B, C, H, W).
            numeric_features (torch.Tensor): Numerical features tensor (B, F).
            text_features (list of str): List of text strings.

        Returns:
            torch.Tensor: Logits for each class.
        """
        # Extract features from the line drawing image
        image_features = self.resnet_backbone(line_image).flatten(1)

        # Process numeric features
        numeric_embeddings = self.numeric_transformer(numeric_features)

        # Ensure text features are a list of strings
        if isinstance(text_features, torch.Tensor):
            text_features = text_features.cpu().numpy().tolist()
            text_features = [str(x) for x in text_features]

        # Tokenize and encode text features using BERT
        encoded_text = self.bert_tokenizer(
            text_features,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(line_image.device)
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']

        # Extract [CLS] token embedding from BERT output
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        text_embeddings = bert_output.last_hidden_state[:, 0, :]

        # Concatenate all feature embeddings
        combined_features = torch.cat((image_features, numeric_embeddings, text_embeddings), dim=1)

        # Pass through fully connected layers with dropout and activation
        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits