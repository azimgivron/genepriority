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

cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_BLAS=ON \
    -DBLA_VENDOR=Generic \
    -DHighFive_DIR=/usr/local \
    -DEigen3_DIR=/usr/include/eigen3 \
    -DBOOST_ROOT=/usr \
    -DPython3_EXECUTABLE=/usr/bin/python3.11 \
    -DPython3_INCLUDE_DIR=/usr/include/python3.11 \
    ..
make -j2
make install

#######################################################
# Set Up Python Virtual Environment
#######################################################
debutSection "SETUP_PYTHON_ENV"
afficheCmd "Creating Python virtual environment"
python3.11 -m venv "$WORK_DIR/.venv"

afficheCmd "Activating virtual environment"
. "$WORK_DIR/.venv/bin/activate"
python3.11 -m pip install --upgrade pip

afficheCmd "Installing SMURFF Python package"
python3.11 -m pip install "$SMURFF_DIR"
finSection "SETUP_PYTHON_ENV"

echo "******************************************"
echo "SMURFF installation completed successfully."
echo "******************************************"
