import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

class FeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = models.resnet50(weights="IMAGENET1K_V2")
        # Get the layers of the model
        self.modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*self.modules)
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 2048  # the length of the feature vector

    def extract_features(self, batch):
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        batch = transform(batch)

        # Pass the image through the Resnet50 model and get the feature maps of the input image
        with torch.no_grad():
            feature = self.model(batch)
            feature = torch.flatten(feature, start_dim=1)

        # Return features to numpy array
        return feature.cpu().detach().numpy()
