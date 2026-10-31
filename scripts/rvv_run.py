# Run one RISC-V binary on an out-of-order core with a chosen vector length (VLEN).
# Usage: build/RISCV/gem5.opt -d <outdir> a5/scripts/rvv_run.py <binary> --vlen 256
import argparse
from m5.objects import RiscvO3CPU
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

p = argparse.ArgumentParser()
p.add_argument("binary")
p.add_argument("--vlen", type=int, default=256)
p.add_argument("--elen", type=int, default=64)
p.add_argument("--l1d", default="32KiB")
a = p.parse_args()


class RVVCore(BaseCPUCore):
    def __init__(self, elen, vlen):
        super().__init__(core=RiscvO3CPU(cpu_id=0), isa=ISA.RISCV)
        self.core.isa[0].elen = elen
        self.core.isa[0].vlen = vlen


board = SimpleBoard(
    clk_freq="2GHz",
    processor=BaseCPUProcessor(cores=[RVVCore(a.elen, a.vlen)]),
    memory=SingleChannelDDR3_1600(),
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1d_size=a.l1d, l1i_size="32KiB", l2_size="512KiB"
    ),
)
board.set_se_binary_workload(BinaryResource(local_path=a.binary))
print("VLEN =", a.vlen, "ELEN =", a.elen)
Simulator(board=board).run()
