#!/bin/zsh
export LIBTORCH=${PWD}/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export RUST_BACKTRACE=1
export RUSTFLAGS="-C target-cpu=native"
cargo run --release --bin yolo_trainer