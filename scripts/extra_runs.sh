#!/bin/bash
set -e
docker run --rm -v "$PWD/a5":/w -w /w debian:trixie bash -c '
  apt-get update -qq && apt-get install -y -qq gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu >/dev/null 2>&1
  CC=riscv64-linux-gnu-gcc
  for N in 1024 4096; do
    $CC -O2 -fno-tree-vectorize -funroll-loops -static -march=rv64gc -DN=$N -DREPS=100 src/daxpy_scalar.c -o bin/daxpy_scalarU_$N
  done
  echo "--- scalar kernel, default build ---"
  riscv64-linux-gnu-objdump -d bin/daxpy_scalar_1024 | grep -B5 -A6 "fmadd.d\|fadd.d" | head -30
'
G=build/RISCV/gem5.opt; S=a5/scripts/rvv_run.py
for N in 1024 4096; do $G -d m5out/a5_scalarU_$N $S a5/bin/daxpy_scalarU_$N --vlen 256 | tail -n 2; done
$G -d m5out/a5_scalar_4096_l1d128 $S a5/bin/daxpy_scalar_4096 --vlen 256 --l1d 128KiB | tail -n 2
for V in 256 512 1024; do $G -d m5out/a5_vec_4096_v${V}_l1d128 $S a5/bin/daxpy_vec_4096 --vlen $V --l1d 128KiB | tail -n 2; done
