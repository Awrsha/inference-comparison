#!/usr/bin/env python3
"""
Script to download pre-trained models and prepare them for inference servers.
"""

import os
import json
import shutil
import torch
import torchvision.models as models
from pathlib import Path

def create_directories():
    """Create necessary directories for the project."""
    dirs = [
        "models",
        "model_repository",
        "model_repository/resnet50/1",
        "model_store",
        "data",
        "app/static/images"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("Created necessary directories.")

def download_imagenet_classes():
    """Download ImageNet class names."""
    try:
        import requests
        url = "https://raw.githubusercontent.com/pytorch/vision/master/torchvision/data/imagenet_classes.txt"
        response = requests.get(url)
        classes = [line.strip() for line in response.text.splitlines()]
        
        # Write to JSON file
        with open("data/imagenet_classes.json", "w") as f:
            json.dump(classes, f)
        
        print("Downloaded ImageNet classes.")
    except Exception as e:
        print(f"Error downloading ImageNet classes: {e}")
        # Create a small sample of classes as fallback
        sample_classes = [f"CLASS_{i}" for i in range(1000)]
        with open("data/imagenet_classes.json", "w") as f:
            json.dump(sample_classes, f)

def download_resnet_pytorch():
    """Download ResNet50 model for PyTorch direct inference."""
    try:
        # Download pre-trained ResNet50
        model = models.resnet50(pretrained=True)
        torch.save(model.state_dict(), "models/resnet50.pth")
        print("Downloaded ResNet50 model for PyTorch.")
    except Exception as e:
        print(f"Error downloading ResNet50 for PyTorch: {e}")

def prepare_torchserve_model():
    """Prepare ResNet50 model for TorchServe."""
    try:
        from torch.package import PackageExporter
        
        # Create simple wrapper for ResNet50
        model = models.resnet50(pretrained=True)
        model.eval()
        
        # Export model to TorchServe format
        model_name = "resnet-50"
        version = "1.0"
        
        # Save model to .mar file (requires torch-model-archiver)
        os.system(f"torch-model-archiver --model-name {model_name} "
                  f"--version {version} "
                  f"--model-file scripts/resnet_handler.py "
                  f"--serialized-file models/resnet50.pth "
                  f"--handler image_classifier "
                  f"--export-path model_store")
        
        print("Prepared ResNet50 model for TorchServe.")
    except Exception as e:
        print(f"Error preparing ResNet50 for TorchServe: {e}")

def prepare_triton_model():
    """Prepare ResNet50 model for Triton Inference Server."""
    try:
        import onnx
        import torch.onnx
        
        model = models.resnet50(pretrained=True)
        model.eval()
        
        # Create example input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        onnx_path = "model_repository/resnet50/1/model.onnx"
        torch.onnx.export(model, dummy_input, onnx_path, 
                          input_names=["input"], 
                          output_names=["output"], 
                          dynamic_axes={"input": {0: "batch_size"}, 
                                      "output": {0: "batch_size"}})
        
        # Create config.pbtxt file for Triton
        config = """
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
"""
        with open("model_repository/resnet50/config.pbtxt", "w") as f:
            f.write(config)
        
        print("Prepared ResNet50 model for Triton Inference Server.")
    except Exception as e:
        print(f"Error preparing ResNet50 for Triton: {e}")

def download_sample_images():
    """Download sample images for testing."""
    try:
        import requests
        from PIL import Image
        from io import BytesIO
        
        # Download a sample image
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg"
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img.save("app/static/images/test_image.jpg")
        
        print("Downloaded sample test image.")
    except Exception as e:
        print(f"Error downloading sample images: {e}")

def create_placeholder_assets():
    """Create placeholder assets for the UI."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create placeholder logo
        logo = Image.new('RGBA', (200, 60), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        draw.text((10, 10), "Inference Compare", fill=(75, 75, 220))
        logo.save("app/static/images/logo.svg")
        
        # Create other placeholder images
        images = {
            "hero-image.svg": (600, 400),
            "triton-logo.png": (120, 80),
            "torchserve-logo.png": (120, 80),
            "pytorch-logo.png": (120, 80),
            "python-logo.png": (120, 80),
            "flask-logo.png": (120, 80),
            "docker-logo.png": (120, 80),
        }
        
        for name, size in images.items():
            img = Image.new('RGB', size, color=(75, 75, 220))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), name.split('.')[0], fill=(255, 255, 255))
            img.save(f"app/static/images/{name}")
        
        print("Created placeholder assets.")
    except Exception as e:
        print(f"Error creating placeholder assets: {e}")

def main():
    print("Starting model download and preparation...")
    create_directories()
    download_imagenet_classes()
    download_resnet_pytorch()
    prepare_torchserve_model()
    prepare_triton_model()
    download_sample_images()
    create_placeholder_assets()
    print("Model download and preparation complete!")

if __name__ == "__main__":
    main()