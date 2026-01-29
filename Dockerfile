# Ubuntu base image - runs natively on ARM64 (Apple Silicon) and x86_64
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Build venv at /opt/venv (outside workspace so it won't be shadowed by mount)
WORKDIR /opt/venv-build
COPY pyproject.toml uv.lock* .python-version ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv python install && \
    uv sync --frozen --no-install-project

RUN uv tool install jupyterlab

# Register the venv Python as a Jupyter kernel
RUN /opt/venv/bin/python -m ipykernel install --name=nanochat --display-name="Python (nanochat)"

WORKDIR /workspace
ENV PATH="/root/.local/share/uv/tools/jupyterlab/bin:/root/.local/bin:/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
