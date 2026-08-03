FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
      torch==2.2.2 torchvision==0.17.2 \
      --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir xquant==2.0.4 \
      --extra-index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
RUN pip install --no-cache-dir \
      numpy==1.26.4 onnx==1.16.2 onnxruntime==1.20.1

WORKDIR /work
