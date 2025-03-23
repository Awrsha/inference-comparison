import os
from pathlib import Path

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'inference-comparision-8930'
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Triton Server Settings
    TRITON_URL = os.environ.get('TRITON_URL') or 'localhost:8000'
    
    # TorchServe Settings
    TORCHSERVE_URL = os.environ.get('TORCHSERVE_URL') or 'localhost:8080'
    
    # Local PyTorch Settings
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}