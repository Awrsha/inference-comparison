import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os

class TorchClient:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load the ResNet50 model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # Define the image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.load_labels()
    
    def load_labels(self):
        # Load ImageNet labels
        labels_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'imagenet_classes.json')
        try:
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
        except:
            self.labels = [f"CLASS_{i}" for i in range(1000)]  # Default labels if file not available
    
    def predict(self, image_path):
        # Open and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Format results
        results = []
        for i in range(5):
            results.append({
                'label': self.labels[top5_idx[i].item()],
                'probability': float(top5_prob[i].item())
            })
        
        return results