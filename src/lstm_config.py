import os
import torch

# --- 路径配置 ---
# 获取当前文件(config.py)的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到 MachineLearningCW 根目录，指向 data
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data') 

# 确保数据目录存在
if not os.path.exists(DATA_DIR):
    print(f"Warning: Data directory not found at {DATA_DIR}")

# --- 模型超参数 ---
BATCH_SIZE = 128        # 批次大小
SEQ_LENGTH = 100        # 输入序列长度 (Time Steps)
HIDDEN_SIZE = 512       # LSTM 隐藏层维度
NUM_LAYERS = 2          # LSTM 层数
DROPOUT = 0.2           # 防止过拟合
LEARNING_RATE = 0.002
EPOCHS = 50

# --- 设备配置 ---
def get_device():
    """自动检测最佳训练设备 (优先支持 Mac M4 MPS)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()