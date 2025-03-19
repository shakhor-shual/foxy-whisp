#!/bin/bash

sudo apt update && sudo apt install -y cmake g++ clang python3-dev libboost-all-dev  patchelf
pip install pybind11
export CMAKE_PREFIX_PATH=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

pip install -r requirements.txt

################### optional ###############################
# uncomnent NEXT before try compile this project to binaries
############################################################

#sudo apt update && sudo apt install -y clang  patchelf
#pip install nuitka


