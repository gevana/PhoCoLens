#FROM nvcr.io/nvidia/pytorch:25.09-py3
#docker pull pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
#FROM nvcr-docker-remote.bart.sec.samsung.net/nvidia/pytorch:24.03-py3
FROM docker-remote.bart.sec.samsung.net/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools wheel
#RUN pip install --no-cache-dir opencv-python-headless
#RUN pip install opencv-python mlflow tensorboard omegaconf imagecodecs rawpy tqdm tifffile Flask-SocketIO ExifRead scikit-image lpips pytorch_lighting libtmux
# Optionally copy your code
# COPY . /workspace

# Optionally install Python packages
COPY requirements_slim.txt .
RUN pip install -r requirements_slim.txt

CMD [ "bash" ]