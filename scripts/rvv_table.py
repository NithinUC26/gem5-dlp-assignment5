# Print a results table from m5out/a5_* runs. Usage: python3 a5/scripts/rvv_table.py 'm5out/a5_*'
import glob, re, sys
def first(pat, txt):
    m = re.search(pat, txt, re.M)
    return float(m.group(1)) if m else None
print(f"{'run':24}{'simSec(us)':>12}{'cycles':>10}{'insts':>10}{'IPC':>7}")
for d in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'm5out/a5_*')):
    try: t = open(d + '/stats.txt').read()
    except OSError: continue
    sec = first(r'^simSeconds\s+([\d.e+-]+)', t)
    cyc = first(r'\.numCycles\s+(\d+)', t)
    ins = first(r'commitStats0\.numInsts\s+(\d+)', t) or first(r'\.committedInsts\s+(\d+)', t)
    if sec is None or cyc is None: continue
    ipc = f"{ins/cyc:7.3f}" if ins else '    n/a'
    print(f"{d.split('/')[-1]:24}{sec*1e6:12.2f}{int(cyc):10d}{int(ins) if ins else 0:10d}{ipc}")
