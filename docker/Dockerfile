# =============================================================================
# Alpamayo-R1 Experiment Runner
# =============================================================================
# Build:  docker build -t alpamayo-runner docker/
# Run:    docker-compose up
# =============================================================================

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# -----------------------------------------------------------------------------
# Build arguments
# -----------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12
ARG USER_ID=1000
ARG GROUP_ID=1000

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \

    # Directories
    APP_DIR=/app \
    DATA_DIR=/data \
    CACHE_DIR=/cache \
    RESULTS_DIR=/results \

    # HuggingFace
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/hub \

    # Torch
    TORCH_HOME=/cache/torch

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    # Build tools
    build-essential \
    git \
    curl \
    wget \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python

# -----------------------------------------------------------------------------
# Create non-root user
# -----------------------------------------------------------------------------
RUN groupadd -g ${GROUP_ID} runner \
    && useradd -m -u ${USER_ID} -g runner runner

# -----------------------------------------------------------------------------
# Create directories
# -----------------------------------------------------------------------------
RUN mkdir -p ${APP_DIR} ${DATA_DIR} ${CACHE_DIR} ${RESULTS_DIR} \
    && chown -R runner:runner ${APP_DIR} ${DATA_DIR} ${CACHE_DIR} ${RESULTS_DIR}

WORKDIR ${APP_DIR}

# -----------------------------------------------------------------------------
# Install Python dependencies (as root for system packages)
# -----------------------------------------------------------------------------
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# -----------------------------------------------------------------------------
# Copy application code
# -----------------------------------------------------------------------------
COPY --chown=runner:runner ../tools/alpamayo/ ${APP_DIR}/alpamayo/
COPY --chown=runner:runner experiments/ ${APP_DIR}/experiments/
COPY --chown=runner:runner docker/scripts/ ${APP_DIR}/scripts/

# Install alpamayo package
RUN pip install -e ${APP_DIR}/alpamayo/

# Make scripts executable
RUN chmod +x ${APP_DIR}/scripts/*.sh

# -----------------------------------------------------------------------------
# Switch to non-root user
# -----------------------------------------------------------------------------
USER runner

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["bash"]
