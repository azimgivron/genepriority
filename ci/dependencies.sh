#!/bin/bash

set -euo pipefail  # Exit immediately on error, undefined variable, or pipe failure

# Update package lists and install dependencies
echo "******************************************"
echo "Installing system dependencies"
echo "******************************************"

# Set non-interactive frontend for package installation
export DEBIAN_FRONTEND=noninteractive

apt update && \
apt install -y --no-install-recommends software-properties-common && \
add-apt-repository -y ppa:deadsnakes/ppa && \
apt update && \
apt install -y --no-install-recommends \
    build-essential \
    cmake \
    git g++ python3.11 \
    libblas-dev \
    liblapack-dev \
    liblapacke-dev \
    libopenmpi-dev \
    openmpi-bin \
    libeigen3-dev \
    libboost-all-dev \
    ca-certificates \
    libhdf5-dev \
    gdb vim wget \
    python3-numpy python3-pybind11 python3-setuptools \
    python3-scipy python3-pandas \
    python3-joblib python3-sklearn \
    python3-h5py \
    python3-pytest \
    python3-parameterized \
    python3-pytest-xdist \
    python3.11-venv \
    python3.11-dev \
    git \
    unzip && \
rm -rf /var/lib/apt/lists/*

# Install Catch v3
echo "******************************************"
echo "Installing Catch2 v3"
echo "******************************************"

cd /tmp && \
    wget -O Catch2.tar.gz https://github.com/catchorg/Catch2/archive/refs/tags/v3.6.0.tar.gz && \
    tar xzf Catch2.tar.gz && \
    rm Catch2.tar.gz && \
    cd Catch2* && \
    cmake -S . -B build -DBUILD_TESTING=OFF && \
    cmake --build build && \
    cmake --install build && \
    cd .. && \
    rm -r Catch2*

# Install HighFive
echo "******************************************"
echo "Installing HighFive"
echo "******************************************"

cd /tmp && \
    wget -O HighFive.tar.gz https://github.com/BlueBrain/HighFive/archive/v2.2.2.tar.gz && \
    tar xzf HighFive.tar.gz && \
    rm HighFive.tar.gz && \
    cd HighFive* && \
    mkdir build && \
    cd build && \
    cmake .. -DHIGHFIVE_USE_BOOST=OFF && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -r HighFive*

echo "******************************************"
echo "Installation completed successfully."
echo "******************************************"
