import torch.nn as nn
import torchvision.models as models

figsize = 64
framesize = 20

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet= models.resnet18(pretrained=False)
        self.fc = nn.Sequential(
            nn.Linear(1000, 5000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 5)
        )
            

    def forward(self, img):
        feat = self.resnet(img)
        return self.fc(feat)
    


            
    
    
