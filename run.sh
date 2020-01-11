#!/bin/zsh
export LIBTORCH=${PWD}/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export RUST_BACKTRACE=1
export RUSTFLAGS="-C target-cpu=native"
#export NUM_CORES=4
#export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
#echo $MKL_NUM_THREADS
#echo $NUM_CORES
#echo $OMP_NUM_THREADS
cargo run --release --bin yolo_trainer