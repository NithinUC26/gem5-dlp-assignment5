#!/bin/bash
# Cross-compile the DAXPY programs for RISC-V Linux inside Docker (run from the a5 folder).
# Debian trixie provides GCC 14, which supports the RISC-V vector intrinsics.
set -e
mkdir -p bin
docker run --rm -v "$PWD":/w -w /w debian:trixie bash -c '
  apt-get update -qq && apt-get install -y -qq gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu >/dev/null
  CC=riscv64-linux-gnu-gcc
  for N in 1024 4096; do
    $CC -O2 -fno-tree-vectorize -static -march=rv64gc  -DN=$N -DREPS=100 src/daxpy_scalar.c -o bin/daxpy_scalar_$N
    $CC -O2 -fno-tree-vectorize -static -march=rv64gcv -DN=$N -DREPS=100 src/daxpy_vec.c    -o bin/daxpy_vec_$N
  done
  riscv64-linux-gnu-objdump -d bin/daxpy_vec_1024    | grep -E "vsetvli|vle64|vfmacc|vse64" | head -8 > bin/vec_inner_loop.txt
  riscv64-linux-gnu-objdump -d bin/daxpy_scalar_1024 | grep -E "fmadd.d|fmul.d|fadd.d|fld|fsd" | head -8 > bin/scalar_inner_loop.txt
'
ls -l bin
echo "--- vector inner loop ---"; cat bin/vec_inner_loop.txt
echo "--- scalar inner loop ---"; cat bin/scalar_inner_loop.txt
