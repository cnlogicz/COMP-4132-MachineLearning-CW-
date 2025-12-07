import torch
import torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.2):
        """
        Standard LSTM architecture for character generation.
        """
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Embedding: Index -> Vector
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 2. LSTM Stack
        # batch_first=True: Input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 3. Dropout & Output Layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden):
        """
        x: (batch_size, seq_len)
        hidden: (h_0, c_0)
        """
        # Embed: (batch, seq) -> (batch, seq, hidden_size)
        x = self.embedding(x)
        
        # LSTM: output -> (batch, seq, hidden_size)
        # hidden -> (h_n, c_n) for next step (if needed)
        out, hidden = self.lstm(x, hidden)
        
        out = self.dropout(out)
        
        # Decode: (batch, seq, hidden) -> (batch, seq, vocab_size)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state and cell state with zeros.
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden