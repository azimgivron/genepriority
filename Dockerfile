FROM --platform=linux/amd64 debian:bookworm

# Set environment variables
ENV HOSTNAME=smurff
ENV DEBIAN_FRONTEND=noninteractive
ENV USERNAME=TheGreatestCoder
ENV PATH="/home/$USERNAME/miniconda3/bin:$PATH"

# Update, install dependencies, create user, and install Oh My Zsh
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv zsh git wget unzip sudo cmake g++ \
    libblas-dev liblapack-dev liblapacke-dev libopenmpi-dev openmpi-bin \
    libeigen3-dev libboost-all-dev ca-certificates libhdf5-dev gdb vim \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd -m -s /bin/zsh $USERNAME && \
    echo "$USERNAME:password" | chpasswd && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    wget https://github.com/ohmyzsh/ohmyzsh/archive/refs/heads/master.zip -O /tmp/ohmyzsh.zip && \
    unzip /tmp/ohmyzsh.zip -d /tmp && \
    mv /tmp/ohmyzsh-master /home/$USERNAME/.oh-my-zsh && \
    cp /home/$USERNAME/.oh-my-zsh/templates/zshrc.zsh-template /home/$USERNAME/.zshrc && \
    rm -rf /tmp/ohmyzsh.zip /tmp/ohmyzsh-master && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.oh-my-zsh /home/$USERNAME/.zshrc && \
    echo "$HOSTNAME" > /etc/hostname

# Switch to the new user
USER $USERNAME

# Install Miniconda and set up conda environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /home/$USERNAME/miniconda3 && \
    rm /tmp/miniconda.sh && \
    echo ". /home/$USERNAME/miniconda3/etc/profile.d/conda.sh" >> /home/$USERNAME/.zshrc && \
    /home/$USERNAME/miniconda3/bin/conda create -y -n thesis-env python=3.9 && \
    echo "conda activate thesis-env" >> /home/$USERNAME/.zshrc && \
    /home/$USERNAME/miniconda3/bin/conda run -n thesis-env pip install \
        numpy \
        scipy \
        pandas \
        scikit-learn \
        pytest \
        pylint \
        docformatter \
        black && \
    /home/$USERNAME/miniconda3/bin/conda run -n thesis-env conda install -c vanderaa smurff -y


# Default shell and command
SHELL ["/bin/zsh", "-c"]
CMD ["zsh"]
