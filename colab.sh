#!/bin/bash
#sudo apt install zsh -y;
#zsh;
export LD_PRELOAD=/usr/lib64-nvidia/libnvidia-ml.so;
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y;
source $HOME/.cargo/env;
rustup install --force nightly-2019-12-25;
rustup default nightly-2019-12-25;
sudo apt install vim -y ;
git clone https://github.com/tiberiusferreira/skynet.git
cd skynet;
wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.3.1.zip;
unzip libtorch-cxx11-abi-shared-with-deps-1.3.1.zip;
wget https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/yolo-v3.ot;
export LD_PRELOAD=/usr/lib64-nvidia/libnvidia-ml.so;
source $HOME/.cargo/env;
./run.sh