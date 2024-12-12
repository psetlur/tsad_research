import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class PositionalEncoding(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        position = torch.arange(0, input_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / encoding_dim))
        pe = torch.zeros(input_size, encoding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]

class CNNEncoder(nn.Module):
    def __init__(self, ts_input_size):
        super().__init__()

        self.ts_input_size = ts_input_size
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=16, kernel_size=3, stride=2, bias=False),
            #nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5, stride=1, dilation=2, bias=False),
            #nn.BatchNorm1d(num_features=8),
        )

        self.positional_encoding = PositionalEncoding(ts_input_size, encoding_dim=512)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 512, self.ts_input_size)
            dummy_output = self.encoder(dummy_input)
            self.encoder_output_size = dummy_output.size(2)
            self.encoder_hidden_size = dummy_output.size(1)
        self.attention = nn.Conv1d(in_channels=self.encoder_hidden_size, out_channels=1, 
                                   kernel_size=1, padding=0, bias=False)
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.encoder_hidden_size, num_heads=4, batch_first=True)

        self.contrastive_enc = nn.Sequential(
            nn.Linear(self.encoder_output_size * self.encoder_hidden_size, 128),
            nn.ReLU(),
            #nn.Dropout(p = 0.1),
            nn.Linear(128, 128),
        )

        for _, m in self.encoder.named_modules():
            if isinstance(m, nn.Conv1d):
                m = spectral_norm(m)
        self.attention = spectral_norm(self.attention)
        for _, m in self.contrastive_enc.named_modules():
            if isinstance(m, nn.Linear):
                m = spectral_norm(m)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1).permute(0, 2, 1)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        attn = F.softmax(self.attention(x), dim = 2)
        x = torch.mul(attn.expand_as(x), x)
        # x = x.permute(0, 2, 1)
        # x, _ = self.multihead_attention(x, x, x)
        # x = x.permute(0, 2, 1)
        x = self.contrastive_enc(x.reshape(x.shape[0], -1))
        return x