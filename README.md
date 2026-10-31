# Assignment 5: Exploring Data-Level Parallelism (DLP)

Code, scripts, and raw results for MSCS-531 (Computer Architecture and Design), Assignment 5.
Author: Nithin Charan Boppana.

The assignment studies how one operation applied to many data items (data-level parallelism) is exploited by vector hardware, SIMD instructions, and parallel loops.

## Contents

| Folder | What it holds |
|---|---|
| `src/` | DAXPY (y = a*x + y, double precision) in plain C (`daxpy_scalar.c`) and with RISC-V vector intrinsics (`daxpy_vec.c`) |
| `scripts/` | gem5 configuration (`rvv_run.py`), build scripts, run scripts, and the results-table printer (`rvv_table.py`) |
| `bin/` | Cross-compiled RISC-V binaries used in the gem5 runs |
| `results/` | gem5 `stats.txt` for every simulation run |
| `simd/` | x86 SIMD benchmark (SSE, AVX2+FMA, AVX-512): SAXPY and dot product, with `simd_final.csv` results |
| `loops/` | Loop-level parallelism experiments with OpenMP, a dependence-analysis example (`detect.c`), and raw run logs |

## 1. Vector architecture (gem5, RISC-V)

- Simulator: gem5 v25.1.0.1, RISC-V build, one out-of-order core at 2 GHz, 32 KiB L1 caches, 512 KiB L2, DDR3-1600.
- Programs: DAXPY, 100 passes over 1,024 or 4,096 doubles. The scalar build uses rv64gc with auto-vectorization off. The vector build uses rv64gcv and RVV intrinsics.
- VLEN swept over 128, 256, 512, and 1,024 bits.
- Extra checks: an unrolled scalar baseline (`-funroll-loops`) and a 128 KiB L1 data cache for the 4,096-element case.

Build and run (from the gem5 directory):

```bash
./a5/scripts/build_riscv.sh      # cross-compile with Docker (debian:trixie, gcc-riscv64-linux-gnu)
./a5/scripts/run_rvv.sh          # gem5 runs: scalar and VLEN 128/256/512/1024
./a5/scripts/extra_runs.sh       # unrolled scalar baseline and 128 KiB L1D runs
python3 a5/scripts/rvv_table.py 'm5out/a5_*'
```

Key findings are in the report:
- Against the default scalar loop, vector DAXPY was 3.75x to 8.41x faster at 1,024 elements.
- That scalar loop was unusually slow (about 9 cycles per element). Against an unrolled scalar loop, the vector gain was 0.75x to 1.67x at 1,024 elements.
- At 4,096 elements with a 32 KiB L1, VLEN scaling stopped. With a 128 KiB L1 it returned, which points to L1 capacity as the limit.

## 2. SIMD extensions (x86-64)

`simd/simd_bench.c` implements SAXPY and a dot product in scalar, SSE, AVX2+FMA, and AVX-512 using compiler intrinsics. The scalar dot product has single-accumulator and four-accumulator versions, and the four-accumulator one is the fair baseline.

```bash
gcc -O2 -mavx512f -mavx2 -mfma -fno-tree-vectorize simd/simd_bench.c -o simd_bench
./simd_bench
```

Requires an x86-64 CPU with AVX-512. Timings come from a shared virtual machine, so run-to-run spread can be large (best-of-N timings are reported). Raw results: `simd/simd_final.csv`.

## 3. Loop-level parallelism

`loops/loops.c` parallelizes four loops with OpenMP and checks each against the serial result. They cover an independent loop, a reduction, a loop-carried dependence (prefix sum, solved with a two-pass algorithm), and loop interchange (matrix multiply). `loops/detect.c` holds loops with and without dependences for compiler dependence reports. Results are in `loops/loops_stats.json` and `loops/run*.txt`.

```bash
gcc -O2 -fopenmp loops/loops.c -o loops_bench && ./loops_bench
```

## Notes and limitations

- The GPU part of the assignment is a written roofline analysis. No GPU measurements were made.
- gem5 instruction counts include program startup, array initialization, and the checksum.
- The cause of the slow default scalar loop in gem5 was not identified.
