FROM --platform=linux/arm64 debian:bookworm

# Install required packages and TensorFlow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git g++ python3 \
    python3-pip && \
    pip install --break-system-packages tensorflow-aarch64==2.16.1

# Use bash shell
SHELL ["/bin/bash", "-c"]

# Expose port for TensorBoard
EXPOSE 6006

# Default command to run TensorBoard
CMD ["tensorboard", "--logdir=/logs", "--host=0.0.0.0", "--port=6006"]
