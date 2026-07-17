# Code Ocean-style environment for the contextual-persistence capsule.
# CPU base is sufficient for the fast path (figures + stats from cached CPF outputs).
# For the full GPU path, swap the base image for a CUDA image, e.g.
#   pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /capsule
COPY environment/requirements.txt /capsule/environment/requirements.txt
RUN pip install --upgrade pip && pip install -r environment/requirements.txt

# install the core CPF package
COPY code/core /capsule/code/core
RUN pip install -e /capsule/code/core

COPY . /capsule
CMD ["bash", "code/run", "figures"]
