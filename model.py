import torch
import torch.nn as nn

class TumorDetector(nn.Module):
    def __init__(self):
        super(TumorDetector, self).__init__()

        self.name = 'TumorDetector'
        self.net = nn.Sequential(
            nn.LazyConv2d(16, 3,1,1), nn.LazyBatchNorm2d(), nn.ReLU(), nn.Dropout(0.4),
            nn.LazyConv2d(16, 3, 1, 1), nn.MaxPool2d(2,2), nn.ReLU(), 
            nn.LazyConv2d(16, 3,1,1), nn.ReLU(), nn.Dropout(0.4),
            nn.LazyConv2d(32, 3, 1, 1), nn.MaxPool2d(2,2), nn.ReLU(), 
            nn.LazyConv2d(32, 3,1,1), nn.ReLU(), nn.Dropout(0.4),
            nn.LazyConv2d(32, 3, 1, 1), nn.MaxPool2d(2,2), nn.ReLU(), 
            nn.Flatten(),
            nn.LazyLinear(512), nn.LazyBatchNorm1d(), nn.ReLU(), nn.Dropout(0.4),
            nn.LazyLinear(512), nn.ReLU(),
            nn.LazyLinear(2)
            )
        
    def forward(self,x):
        return self.net(x)