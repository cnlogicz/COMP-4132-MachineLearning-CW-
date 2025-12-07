import torch
import torch.nn.functional as F
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import lstm_config as config
from model import CharLSTM
from data_loader import LOTRDataset

def generate_text(model, dataset, start_string="Frodo", generation_length=500, temperature=0.8):
    model.eval()
    
    # 注意：如果遇到没见过的字符，这里默认给0或者跳过，为了简单我们忽略未知字符
    input_indices = [dataset.char2id[c] for c in start_string if c in dataset.char2id]
    
    if not input_indices:
        print("Error: Start string contains no valid characters from vocabulary.")
        return

    input_tensor = torch.tensor(input_indices).unsqueeze(0).to(config.DEVICE)
    
    hidden = model.init_hidden(1, config.DEVICE)
    
    generated_text = start_string
    print(f"--- Generating (Temp: {temperature}) ---")
    print(start_string, end="", flush=True)

    with torch.no_grad():
        # Warm-up
        for i in range(len(input_indices) - 1):
            _, hidden = model(input_tensor[:, i:i+1], hidden)
        
        last_char_idx = input_tensor[:, -1].unsqueeze(1)
        
        for i in range(generation_length):
            output, hidden = model(last_char_idx, hidden)
            output_logits = output.squeeze()
            
            # 温度采样
            output_logits = output_logits / temperature
            probs = F.softmax(output_logits, dim=0)
            
            predicted_idx = torch.multinomial(probs, 1).item()
            
            predicted_char = dataset.id2char[predicted_idx]
            
            print(predicted_char, end="", flush=True)
            generated_text += predicted_char
            
            last_char_idx = torch.tensor([[predicted_idx]]).to(config.DEVICE)
            
    print("\n" + "="*50)
    return generated_text

def main():
    print(f"Loading data and model on {config.DEVICE}...")
    
    # 这会重新读取数据并构建字典，确保映射与训练时一致（假设数据文件没变）
    dataset = LOTRDataset(config.DATA_DIR, config.SEQ_LENGTH)
    vocab_size = dataset.vocab_size
    
    model = CharLSTM(
        vocab_size=vocab_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS
    ).to(config.DEVICE)
    
    model_path = os.path.join(config.BASE_DIR, "best_lstm_model00005.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"[Warning] Model file not found at {model_path}")
        print("Using untrained model (output will be random garbage).")
    
    prompt = "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone,"
    generate_text(model, dataset, start_string=prompt, generation_length=500, temperature=0.5)

if __name__ == "__main__":
    main()