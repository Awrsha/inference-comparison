import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from io import BytesIO
import json
import os

class TritonClient:
    def __init__(self, url="localhost:8000"):
        self.url = url
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = "resnet50"
        self.load_labels()
    
    def load_labels(self):
        # Load ImageNet labels
        labels_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'imagenet_classes.json')
        try:
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
        except:
            self.labels = [f"CLASS_{i}" for i in range(1000)]  # Default labels if file not available
    
    def preprocess_image(self, image_path):
        # Open the image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224
        img = img.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_array = (img_array - mean) / std
        
        # Transpose from (H, W, C) to (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path):
        # Preprocess the image
        input_data = self.preprocess_image(image_path)
        
        # Create inference request
        inputs = []
        inputs.append(httpclient.InferInput("input", input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("output"))
        
        # Run inference
        response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        
        # Get results
        output_data = response.as_numpy("output")
        
        # Post-process results
        probabilities = np.exp(output_data) / np.sum(np.exp(output_data), axis=1, keepdims=True)
        
        # Get top 5 predictions
        top5_idx = np.argsort(probabilities[0])[-5:][::-1]
        top5_probs = probabilities[0][top5_idx]
        
        # Format results
        results = []
        for i in range(5):
            results.append({
                'label': self.labels[top5_idx[i]],
                'probability': float(top5_probs[i])
            })
        
        return results