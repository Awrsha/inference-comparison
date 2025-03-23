from app import create_app
import os

# Ensure upload directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), 'app', 'static', 'uploads'), exist_ok=True)

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)