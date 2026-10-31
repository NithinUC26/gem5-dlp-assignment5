#!/bin/bash
# Run from the gem5 root folder. Runs scalar and vector DAXPY at several vector lengths.
G=build/RISCV/gem5.opt
S=a5/scripts/rvv_run.py
for N in 1024 4096; do
  $G -d m5out/a5_scalar_$N $S a5/bin/daxpy_scalar_$N --vlen 256 | tail -n 3
  for V in 128 256 512 1024; do
    $G -d m5out/a5_vec_${N}_v$V $S a5/bin/daxpy_vec_$N --vlen $V | tail -n 3
  done
done
