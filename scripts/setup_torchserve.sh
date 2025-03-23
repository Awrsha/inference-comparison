#!/bin/bash
# Setup script for TorchServe

# Check if model store is empty
if [ -z "$(ls -A /home/model-server/model-store 2>/dev/null)" ]; then
    echo "Model store is empty. Setting up default models..."
    
    # Install packages
    pip install torch torchvision torch-model-archiver
    
    # Create and archive a model
    cd /tmp
    cat > handler.py << EOF
from torchvision import models, transforms
from PIL import Image
import torch
import io
import json

class ModelHandler:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load class labels
        with open('imagenet_classes.json', 'r') as f:
            self.classes = json.load(f)
    
    def preprocess(self, data):
        image = Image.open(io.BytesIO(data))
        image = self.transform(image)
        return image.unsqueeze(0)
    
    def inference(self, data):
        with torch.no_grad():
            output = self.model(data)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities
    
    def postprocess(self, data):
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(data, 5)
        
        result = []
        for i in range(top5_idx.size(0)):
            result.append({
                'class': top5_idx[i].item(),
                'probability': top5_prob[i].item()
            })
        
        return result
EOF

    # Download ImageNet classes
    curl -O https://raw.githubusercontent.com/pytorch/vision/master/torchvision/data/imagenet_classes.txt
    python -c "import json; classes = [line.strip() for line in open('imagenet_classes.txt').readlines()]; json.dump(classes, open('imagenet_classes.json', 'w'))"
    
    # Create a model archive
    torch-model-archiver --model-name resnet-50 \
                          --version 1.0 \
                          --handler handler.py \
                          --runtime python \
                          --export-path /home/model-server/model-store
    
    # Copy the classes file
    cp imagenet_classes.json /home/model-server/
fi

# Create config file if it doesn't exist
if [ ! -f /home/model-server/config.properties ]; then
    echo "Creating TorchServe config file..."
    cat > /home/model-server/config.properties << EOF
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=4
job_queue_size=10
model_store=/home/model-server/model-store
load_models=resnet-50.mar
EOF
fi

# Start TorchServe
echo "Starting TorchServe..."
torchserve --start --model-store /home/model-server/model-store --ncs