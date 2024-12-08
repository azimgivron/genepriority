#!/bin/bash

# Utility Functions
function afficheCmd() {
    local cmd="${1}"
    echo -e "\e[33m  \$> ${cmd} \e[0m"
}

function debutSection() {
    local titre="${1}"
    echo -e "\e[0Ksection_start:$(date +%s):${titre}[collapsed=true]\r\e[0K${titre}"
}

function finSection() {
    local titre="${1}"
    echo -e "\e[0Ksection_end:$(date +%s):${titre}\r\e[0K"
}

# Set non-interactive frontend for package installation
export DEBIAN_FRONTEND=noninteractive

# Script Context
echo "******************************************"
echo "CONTEXT"
echo "******************************************"
echo "SCRIPT:     ${0}"
echo "HOSTNAME:   ${HOSTNAME}"
echo "PWD:        ${PWD}"
echo "DATE:       $(date)"
echo "SHELL:      ${SHELL}"
echo "PATH:       ${PATH}"
echo "******************************************"

# Variables
TEMP_DIR=$(mktemp -d)
WORK_DIR="/app/work"
SMURFF_REPO_URL="https://github.com/ExaScience/smurff.git"
PYTHON_VERSION="3.11"
COMMIT="35f766e8fa5d96fef90a943f6ef256dc5d5820e2"

#######################################################
# Clone SMURFF Repository
#######################################################
debutSection "DOWNLOAD_SMURFF"
afficheCmd "Cloning SMURFF repository from GitHub"
git clone "$SMURFF_REPO_URL" "$TEMP_DIR/smurff"
cd "$TEMP_DIR/smurff"

afficheCmd "Checking out specific commit: $COMMIT"
git checkout "$COMMIT"

SMURFF_DIR="$TEMP_DIR/smurff"
BUILD_DIR="$SMURFF_DIR/build"

finSection "DOWNLOAD_SMURFF"

#######################################################
# Build and Install SMURFF
#######################################################
debutSection "BUILD_AND_INSTALL_SMURFF"
afficheCmd "Setting up build directory"
mkdir -p "$BUILD_DIR" "$WORK_DIR"

cd "$TEMP_DIR"
afficheCmd "Running CMake configuration"
cmake -S "$SMURFF_DIR" -B "$BUILD_DIR"

afficheCmd "Building SMURFF"
cmake --build "$BUILD_DIR"

afficheCmd "Installing SMURFF"
cmake --install "$BUILD_DIR"

afficheCmd "Running SMURFF self-tests"
"$BUILD_DIR/bin/smurff" --bist
finSection "BUILD_AND_INSTALL_SMURFF"

#######################################################
# Set Up Python Virtual Environment
#######################################################
debutSection "SETUP_PYTHON_ENV"
afficheCmd "Creating Python virtual environment"
python3 -m venv "$WORK_DIR/.venv"

afficheCmd "Activating virtual environment"
source "$WORK_DIR/.venv/bin/activate"

afficheCmd "Installing SMURFF Python package"
pip install -v "$SMURFF_DIR"
echo "source $WORK_DIR/.venv/bin/activate" >> /home/root/.bashrc
finSection "SETUP_PYTHON_ENV"

echo "******************************************"
echo "SMURFF installation completed successfully."
echo "******************************************"
