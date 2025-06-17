FROM python:3.10-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY requirements/base.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install django uvicorn

COPY . .

# Create necessary directories for logs and data
RUN mkdir -p /app/data/logs && \
    mkdir -p /app/data/artifacts && \
    chmod 755 /app/data/logs

# Make sure the core and ml packages are in Python path
ENV PYTHONPATH=/app

CMD ["python", "core/manage.py", "runserver", "0.0.0.0:8000"]
# lub jeśli chcesz ASGI:
# CMD ["uvicorn", "core.config.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
