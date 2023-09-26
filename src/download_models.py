'''
This file is just used to download the models from the internet. 
'''
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch

def main():
    try:
        print("Downloading EfficientNet-B3...")
        _ = EfficientNet.from_pretrained("efficientnet-b3")
    except Exception as e:
        print(f"Error while downloading EfficientNet-B3: {e}")
        
    try:
        print("Downloading EfficientNet-B4...")
        _ = EfficientNet.from_pretrained("efficientnet-b4")
    except Exception as e:
        print(f"Error while downloading EfficientNet-B4: {e}")
        
    try:
        print("Downloading vgg19...")
        _ = models.vgg19()
    except Exception as e:
        print(f"Error while downloading vgg19: {e}")
        
    try:
        print("Downloading resnet50...")
        _ = models.resnet50()
    except Exception as e:
        print(f"Error while downloading resnet50: {e}")
    
    try:
        print("Downloading dinov2_vits14...")
        _ = torch.hub.load('facebookresearch/dinov2', "dinov2_vits14")
    except Exception as e:
        print(f"Error while downloading dinov2_vits14: {e}")
    
    
if __name__ == "__main__":
    main()