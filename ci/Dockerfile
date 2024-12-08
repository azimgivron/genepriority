# Use Ubuntu 20.04
FROM --platform=linux/amd64 ubuntu:20.04

# Set non-interactive mode for package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy installation scripts
COPY dependencies.sh /app/dependencies.sh
COPY install_smurff.sh /app/install_smurff.sh

# Run installation scripts
RUN chmod +x /app/dependencies.sh /app/install_smurff.sh && \
    /app/dependencies.sh && \
    /app/install_smurff.sh

# Set entry point
ENTRYPOINT ["/bin/bash"]
