import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # Cargar el modelo ResNet18 preentrenado
        self.resnet = models.resnet18(pretrained=True)
        
        # Reemplazar la última capa totalmente conectada para que tenga el número de clases adecuado
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
