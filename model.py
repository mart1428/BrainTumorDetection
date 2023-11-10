import torch
import torch.nn as nn
import torchvision.models as models

class TumorDetector(nn.Module):
    def __init__(self):
        super(TumorDetector, self).__init__()

        self.name = 'Tumor Detector'
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
    
class TumorDetectorResNet(nn.Module):
    def __init__(self):
        super(TumorDetectorResNet, self).__init__()

        self.name = 'Tumor Detector ResNet'
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(nn.Linear(512, 2048), nn.Dropout(0.5), nn.ReLU(),
                                       nn.Linear(2048, 2048), nn.Dropout(0.5), nn.ReLU(),
                                       nn.Linear(2048, 2048), nn.Dropout(0.5), nn.ReLU(),
                                       nn.Linear(2048,2))
        self.resnet.fc.requires_grad_ = True


    def forward(self, x):
        x = self.resnet(x)
        return x