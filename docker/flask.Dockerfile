FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required directories
RUN mkdir -p app/static/uploads
RUN mkdir -p app/static/images
RUN mkdir -p models
RUN mkdir -p data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=run.py
ENV FLASK_ENV=development

# Download models and data
RUN python scripts/download_models.py

CMD ["python", "run.py"]