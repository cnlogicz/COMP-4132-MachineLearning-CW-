import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time 

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import lstm_config as config
    from data_loader import get_loaders_and_vocab
    from model import CharLSTM
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# === 日志记录辅助类 ===
class Logger(object):
    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        # 使用 'a' 模式 (append) 防止覆盖，encoding='utf-8' 防止乱码
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        # 写入终端
        self.terminal.write(message)
        # 写入文件
        self.log.write(message)
        self.log.flush() # 立即刷新缓冲区，确保程序崩溃时日志也能保存

    def flush(self):
        # 兼容性函数，某些库需要调用 flush
        self.terminal.flush()
        self.log.flush()

def train():
    # 1. 准备数据
    print(f"--- [Setup] Device: {config.DEVICE} ---")
    
    train_loader, val_loader, dataset = get_loaders_and_vocab()
    
    vocab_size = dataset.vocab_size
    print(f"Data loaded. Vocab size: {vocab_size}")

    # 2. 初始化模型
    model = CharLSTM(
        vocab_size=vocab_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
    )

    # 4. 训练循环
    best_val_loss = float('inf')
    
    print("Starting training...")
    
    for epoch in range(config.EPOCHS):
        model.train() # 切换到训练模式
        train_loss = 0
        
        # --- Training Step ---
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            hidden = model.init_hidden(config.BATCH_SIZE, config.DEVICE)
            model.zero_grad()
            
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            train_loss += loss.item()
            
            # 打印进度
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{config.EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # 每 500 步保存一个"最新模型"用于测试
            if (i+1) % 500 == 0:
                temp_save_path = os.path.join(config.BASE_DIR, "latest_model.pth")
                torch.save(model.state_dict(), temp_save_path)
                print(f"    >>> [Mid-Epoch Save] Latest checkpoint saved to {temp_save_path} (Step {i+1})")
                
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Step ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                hidden = model.init_hidden(config.BATCH_SIZE, config.DEVICE)
                output, hidden = model(inputs, hidden)
                loss = criterion(output, targets.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss) # 根据验证集 Loss 决定是否降低学习率

        print(f"==> Epoch {epoch+1} Report: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config.BASE_DIR, "best_lstm_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    [Checkpoint] Model saved to {save_path}")

if __name__ == "__main__":
    # === 设置日志保存路径 ===
    # 1. 创建 logs 文件夹 (如果不存在)
    log_dir = os.path.join(config.BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. 生成带时间戳的文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"train_log_{timestamp}.txt")
    print(f"Logging output to: {log_filename}")
    
    # 4. 劫持 sys.stdout
    # 从这一行开始，所有的 print() 都会同时写入文件
    sys.stdout = Logger(log_filename)
    
    # 运行训练
    train()