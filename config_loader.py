import yaml
import os


def check_config_infer(config):
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


# --------------------------
# Load Configuration with Validation
# --------------------------
def load_config_infer(config_path='./config/config_inference.yaml'):
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



