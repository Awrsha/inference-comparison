#!/bin/bash
# Setup script for Triton Inference Server

# Check if model repository is empty
if [ -z "$(ls -A /models/resnet50 2>/dev/null)" ]; then
    echo "Model repository is empty. Setting up default models..."
    
    # Create model repository structure
    mkdir -p /models/resnet50/1
    
    # Copy configuration if it doesn't exist
    if [ ! -f /models/resnet50/config.pbtxt ]; then
        echo "Creating model configuration..."
        cat > /models/resnet50/config.pbtxt << EOF
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
EOF
    fi
    
    # Download a pre-trained model if it doesn't exist
    if [ ! -f /models/resnet50/1/model.onnx ]; then
        echo "Downloading pre-trained ResNet50 model..."
        apt-get update && apt-get install -y python3-pip
        pip install torch torchvision onnx
        
        python3 - << EOF
import torch
import torchvision.models as models
model = models.resnet50(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, '/models/resnet50/1/model.onnx', 
                  input_names=["input"], 
                  output_names=["output"], 
                  dynamic_axes={"input": {0: "batch_size"}, 
                               "output": {0: "batch_size"}})
print("ONNX model saved to /models/resnet50/1/model.onnx")
EOF
    fi
fi

# Start Triton server
echo "Starting Triton Inference Server..."
tritonserver --model-repository=/models --log-verbose=1