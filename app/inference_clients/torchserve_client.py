import requests
import json
from PIL import Image
import os

class TorchServeClient:
    def __init__(self, url="localhost:8080"):
        self.url = url
        self.model_name = "resnet-50"
        self.endpoint = f"http://{url}/predictions/{self.model_name}"
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
        # Open the image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Send the request to TorchServe
        response = requests.post(
            self.endpoint,
            data=image_data,
            headers={'Content-Type': 'application/octet-stream'}
        )
        
        # Parse the response
        if response.status_code != 200:
            raise Exception(f"TorchServe request failed with status {response.status_code}: {response.text}")
        
        response_json = response.json()
        
        # Format results
        results = []
        for item in response_json:
            idx = int(item['class'])
            prob = float(item['probability'])
            results.append({
                'label': self.labels[idx] if idx < len(self.labels) else f"CLASS_{idx}",
                'probability': prob
            })
        
        # Ensure we have at most 5 predictions
        return results[:5]