from torchvision import transforms
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch
from CustomModels import DinoVisionClassifier

classes = {0: 'Glas', 1: 'Organic', 2: 'Papier', 3: 'Restmüll', 4: 'Wertstoff'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ]
)

transform_dinov2 = transforms.Compose(
    [   transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ]
)

def load_specific_model(model_name):
    current_model = None
    if model_name == "EfficientNet-B3":
        current_model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=len(classes.keys()))
        current_model.load_state_dict(torch.load("src/models/eff_b3_model.pt", map_location="cpu"))
    elif model_name == "EfficientNet-B4":
        current_model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=len(classes.keys()))
        current_model.load_state_dict(torch.load("src/models/eff_b4.pt", map_location="cpu"))
    elif model_name == "vgg19":
        current_model = models.vgg19()
        in_features = current_model.classifier[0].in_features
        current_model.classifier = torch.nn.Linear(in_features, len(classes.keys()))
        current_model.load_state_dict(torch.load("src/models/vgg19.pt", map_location="cpu"))
    elif model_name == "resnet50":
        current_model = models.resnet50()
        in_features = current_model.fc.in_features
        current_model.fc = torch.nn.Linear(in_features, len(classes.keys()))
        current_model.load_state_dict(torch.load("src/models/resnet50.pt", map_location="cpu"))
    elif model_name == "dinov2_vits14":
        current_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vits14")
        current_model = DinoVisionClassifier(current_model, num_classes=len(classes.keys()))
        current_model.load_state_dict(torch.load("src/models/dinov2_vits14_0.054_98.00.pth", map_location="cpu"))
    
    print(f"Loaded model {model_name}")
    return current_model.eval().to(device)
    
def inference(model, inp):
    model.eval()
    inp = transform(inp) if model.__class__.__name__ != "DinoVisionClassifier" else transform_dinov2(inp)
    inp = inp.unsqueeze(0).to(device)
    if torch.cuda.is_available():
        with torch.no_grad(), torch.cuda.amp.autocast():
            prediction = torch.nn.functional.softmax(model(inp)[0], dim=0).cpu().numpy()
    else:
        with torch.no_grad():
            prediction = torch.nn.functional.softmax(model(inp)[0], dim=0).cpu().numpy()
        
    confidences = {classes[i]: float(prediction[i]) for i in range(len(classes.keys()))}
    return confidences
    