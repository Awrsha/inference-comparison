import os
import time
import uuid
from flask import Blueprint, render_template, request, jsonify, current_app, url_for
from werkzeug.utils import secure_filename
from app.inference_clients import TritonClient, TorchClient, TorchServeClient

main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)

# Initialize clients
triton_client = None
torch_client = None
torchserve_client = None

@main_bp.before_app_first_request
def initialize_clients():
    global triton_client, torch_client, torchserve_client
    triton_client = TritonClient(current_app.config['TRITON_URL'])
    torch_client = TorchClient(current_app.config['MODEL_DIR'])
    torchserve_client = TorchServeClient(current_app.config['TORCHSERVE_URL'])

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def save_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        return file_path
    return None

# Routes
@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@main_bp.route('/comparison')
def comparison():
    return render_template('comparison.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

# API Endpoints
@api_bp.route('/inference/triton', methods=['POST'])
def inference_triton():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = save_file(file)
    
    if not file_path:
        return jsonify({'error': 'Invalid file'}), 400
    
    start_time = time.time()
    try:
        result = triton_client.predict(file_path)
        inference_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'result': result,
            'inference_time': inference_time,
            'server': 'Triton Inference Server'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/inference/pytorch', methods=['POST'])
def inference_pytorch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = save_file(file)
    
    if not file_path:
        return jsonify({'error': 'Invalid file'}), 400
    
    start_time = time.time()
    try:
        result = torch_client.predict(file_path)
        inference_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'result': result,
            'inference_time': inference_time,
            'server': 'PyTorch Direct'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/inference/torchserve', methods=['POST'])
def inference_torchserve():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = save_file(file)
    
    if not file_path:
        return jsonify({'error': 'Invalid file'}), 400
    
    start_time = time.time()
    try:
        result = torchserve_client.predict(file_path)
        inference_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'result': result,
            'inference_time': inference_time,
            'server': 'TorchServe'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/inference/all', methods=['POST'])
def inference_all():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = save_file(file)
    
    if not file_path:
        return jsonify({'error': 'Invalid file'}), 400
    
    results = {}
    
    # Triton Inference
    try:
        start_time = time.time()
        triton_result = triton_client.predict(file_path)
        triton_time = time.time() - start_time
        results['triton'] = {
            'success': True,
            'result': triton_result,
            'inference_time': triton_time
        }
    except Exception as e:
        results['triton'] = {
            'success': False,
            'error': str(e)
        }
    
    # PyTorch Direct
    try:
        start_time = time.time()
        torch_result = torch_client.predict(file_path)
        torch_time = time.time() - start_time
        results['pytorch'] = {
            'success': True,
            'result': torch_result,
            'inference_time': torch_time
        }
    except Exception as e:
        results['pytorch'] = {
            'success': False,
            'error': str(e)
        }
    
    # TorchServe
    try:
        start_time = time.time()
        torchserve_result = torchserve_client.predict(file_path)
        torchserve_time = time.time() - start_time
        results['torchserve'] = {
            'success': True,
            'result': torchserve_result,
            'inference_time': torchserve_time
        }
    except Exception as e:
        results['torchserve'] = {
            'success': False,
            'error': str(e)
        }
    
    return jsonify(results)

@api_bp.route('/benchmark', methods=['POST'])
def benchmark():
    """Run benchmark with multiple iterations"""
    iterations = int(request.json.get('iterations', 10))
    server_type = request.json.get('server', 'all')
    
    benchmark_results = {
        'triton': {'times': [], 'avg': 0, 'errors': 0},
        'pytorch': {'times': [], 'avg': 0, 'errors': 0},
        'torchserve': {'times': [], 'avg': 0, 'errors': 0}
    }
    
    # Example test image path - in production, allow uploading or use a test set
    test_image_path = os.path.join(current_app.config['BASE_DIR'], 'app', 'static', 'images', 'test_image.jpg')
    
    if not os.path.exists(test_image_path):
        return jsonify({'error': 'Test image not found'}), 404
    
    for i in range(iterations):
        # Triton benchmark
        if server_type == 'all' or server_type == 'triton':
            try:
                start_time = time.time()
                triton_client.predict(test_image_path)
                inference_time = time.time() - start_time
                benchmark_results['triton']['times'].append(inference_time)
            except Exception:
                benchmark_results['triton']['errors'] += 1
        
        # PyTorch benchmark
        if server_type == 'all' or server_type == 'pytorch':
            try:
                start_time = time.time()
                torch_client.predict(test_image_path)
                inference_time = time.time() - start_time
                benchmark_results['pytorch']['times'].append(inference_time)
            except Exception:
                benchmark_results['pytorch']['errors'] += 1
        
        # TorchServe benchmark
        if server_type == 'all' or server_type == 'torchserve':
            try:
                start_time = time.time()
                torchserve_client.predict(test_image_path)
                inference_time = time.time() - start_time
                benchmark_results['torchserve']['times'].append(inference_time)
            except Exception:
                benchmark_results['torchserve']['errors'] += 1
    
    # Calculate averages
    for server in benchmark_results:
        times = benchmark_results[server]['times']
        if times:
            benchmark_results[server]['avg'] = sum(times) / len(times)
    
    return jsonify(benchmark_results)