FROM --platform=linux/arm64 debian:bookworm

# Set environment variables
ENV HOSTNAME=thesis-server
ENV DEBIAN_FRONTEND=noninteractive
ENV USERNAME=TheGreatestCoder
ENV VENV_NAME=NEGradient_GenePriority_venv
ENV VENV_PATH="/home/$USERNAME/$VENV_NAME"
ENV PATH="$VENV_PATH/bin:$PATH"

# Updates, package installation, Catch2, HighFive, and user setup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git g++ python3 \
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
        python3-pip \
        python3-numpy python3-pybind11 python3-setuptools \
        python3-scipy python3-pandas \
        python3-joblib python3-sklearn \
        python3-h5py \
        python3-pytest \
        python3-parameterized \
        python3-pytest-xdist \
        python3.11-venv \
        unzip \
        sudo \
        zsh \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp && \
    wget -O Catch2.tar.gz https://github.com/catchorg/Catch2/archive/refs/tags/v3.6.0.tar.gz && \
    tar xzf Catch2.tar.gz && \
    rm Catch2.tar.gz && \
    cd Catch2* && \
    cmake -S . -B build  -DBUILD_TESTING=OFF && \
    cmake --build build && \
    cmake --install build && \
    cd .. && \
    rm -rf Catch* && \
    wget -O HighFive.tar.gz https://github.com/BlueBrain/HighFive/archive/v2.2.2.tar.gz && \
    tar xzf HighFive.tar.gz && \
    rm HighFive.tar.gz && \
    cd HighFive* && \
    mkdir build && \
    cd build && \
    cmake .. -DHIGHFIVE_USE_BOOST=OFF && \
    make -j2 && \
    make install && \
    cd ../.. && \
    rm -rf HighFive* && \
    useradd -m -s /bin/zsh $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    sh -c "echo '$HOSTNAME' > /etc/hostname" && \
    wget https://github.com/ohmyzsh/ohmyzsh/archive/refs/heads/master.zip -O /tmp/ohmyzsh.zip && \
    unzip /tmp/ohmyzsh.zip -d /tmp && \
    mv /tmp/ohmyzsh-master /home/$USERNAME/.oh-my-zsh && \
    cp /home/$USERNAME/.oh-my-zsh/templates/zshrc.zsh-template /home/$USERNAME/.zshrc && \
    rm -rf /tmp/ohmyzsh.zip /tmp/ohmyzsh-master && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.oh-my-zsh /home/$USERNAME/.zshrc && \
    cd /tmp && \
    git clone https://github.com/ExaScience/smurff.git && \
    cd smurff && \
    git checkout c5c5d50fdc7f1663bb972c75d277a10ebc47db59 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_BLAS=ON \
          -DBLA_VENDOR=Generic \
          -DHighFive_DIR=/usr/local \
          -DEigen3_DIR=/usr/include/eigen3 \
          -DBOOST_ROOT=/usr \
          .. && \
    make -j2 && \
    make install

# Switch to the new user
USER $USERNAME

# Copy requirements.txt to the user's home directory
COPY --chown=$USERNAME:$USERNAME requirements.txt /home/$USERNAME/requirements.txt

# Set up Python environment and install libraries
RUN python3 -m venv $VENV_PATH && \
    $VENV_PATH/bin/pip install --upgrade pip && \
    $VENV_PATH/bin/pip install -r /home/$USERNAME/requirements.txt && \
    sudo $VENV_PATH/bin/pip install /tmp/smurff && \
    sudo rm -rf /tmp/smurff && \
    echo "source $VENV_PATH/bin/activate" >> /home/$USERNAME/.zshrc

# Default shell and command
SHELL ["/bin/zsh", "-c"]
CMD ["zsh"]
