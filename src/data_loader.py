import torch
import glob
import os
import sys
from collections import Counter # 引入 Counter 
from torch.utils.data import Dataset, DataLoader, random_split

# 路径修复逻辑
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import lstm_config as config
except ImportError:
    from . import lstm_config as config

class LOTRDataset(Dataset): # 对应 MyDataset 类 
    def __init__(self, data_dir, seq_length):
        self.data_dir = data_dir
        self.seq_length = seq_length
        
        #  初始化时直接调用加载和处理函数 
        self.listOfChars = self.loadData()           # 对应 self.listOfWords
        self.listOfUniqueChars = self.obtainUniqueChars() # 对应 self.listOfUniqueWords
        
        # 构建 id2char (id2word) 
        self.id2char = {i: c for i, c in enumerate(self.listOfUniqueChars)}
        
        # 构建 char2id (word2id) 
        self.char2id = {c: i for i, c in enumerate(self.listOfUniqueChars)}
        
        # 将文本转换为 ID 列表 
        self.listOfIds = [self.char2id[c] for c in self.listOfChars]
        
        # 额外属性：词汇表大小
        self.vocab_size = len(self.listOfUniqueChars)

    def loadData(self): # 对应 loadWords 
        """
        读取并合并所有 txt 文件字符。
        Lab 4 使用 pd.read_csv，我们这里适配为读取 txt，但结构一致。
        """
        txt_files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.data_dir}")
            
        full_text = ""
        # 保持之前的健壮性读取逻辑
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for fpath in txt_files:
            for enc in encodings:
                try:
                    with open(fpath, 'r', encoding=enc) as f:
                        full_text += f.read() + "\n"
                    break
                except: continue
        
        # 返回字符列表 (Lab 4 返回的是 split 后的单词列表) 
        return list(full_text)

    def obtainUniqueChars(self): # 对应 obtainUniqueWords 
        """
        使用 Counter 统计词频并排序
        """
        # 使用 Counter 统计频率 
        charCounts = Counter(self.listOfChars)
        
        # 按频率降序排列 (出现最多的字符 ID 为 0) 
        return sorted(charCounts, key=charCounts.get, reverse=True)

    def __len__(self): 
        return len(self.listOfIds) - self.seq_length

    def __getitem__(self, index): 
        #  直接切片并转换为 Tensor 返回
        input_seq = torch.tensor(self.listOfIds[index : index + self.seq_length])
        target_seq = torch.tensor(self.listOfIds[index + 1 : index + self.seq_length + 1])
        return input_seq, target_seq

# --- 辅助函数：适配 train.py 调用 ---
def get_loaders_and_vocab():
    """
    为了方便 train.py 调用，封装加载逻辑
    """
    dataset = LOTRDataset(config.DATA_DIR, config.SEQ_LENGTH)
    
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_data, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=0 # Windows/Mac 兼容性
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        drop_last=True,
        num_workers=0
    )
    
    # 返回 dataset 实例以便获取 vocab_size 和映射表
    return train_loader, val_loader, dataset

if __name__ == "__main__":
    # 测试代码
    try:
        print("Testing Dataset...")
        loader, _, ds = get_loaders_and_vocab()
        print(f"Vocab Size (Unique Chars): {ds.vocab_size}")
        print(f"Most common char (ID 0): '{ds.id2char[0]}'") # 应该是空格或e
        inputs, targets = next(iter(loader))
        print(f"Input shape: {inputs.shape}")
        print("Test Passed.")
    except Exception as e:
        print(f"Error: {e}")