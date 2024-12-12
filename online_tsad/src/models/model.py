import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# this is slow - try removing batch norm and dropout and see
class CNNEncoder(nn.Module):
    def __init__(self, ts_input_size):
        super().__init__()

        self.ts_input_size = ts_input_size
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2, bias=False),
            #nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, stride=1, dilation=2, bias=False),
            #nn.BatchNorm1d(num_features=8),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.ts_input_size)
            dummy_output = self.encoder(dummy_input)
            self.encoder_output_size = dummy_output.size(2)
            self.encoder_hidden_size = dummy_output.size(1)
        self.attention = nn.Conv1d(in_channels=self.encoder_hidden_size, out_channels=1, 
                                   kernel_size=1, padding=0, bias=False)

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
        x = self.encoder(x)
        attn = F.softmax(self.attention(x), dim = 2)
        x = torch.mul(attn.expand_as(x), x)
        x = self.contrastive_enc(x.reshape(x.shape[0], -1))
        return x