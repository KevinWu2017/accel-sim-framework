# WeightAddress/HBF port for Accel-Sim v2.0.1

This branch reimplements the functional WeightAddress path from the older HBF
fork on top of upstream Accel-Sim v2.0.1 and GPGPU-Sim v4.2.0.

## Trace generation

The capture script writes a `weightsAddress` map alongside raw NVBit traces.
Each non-comment map line starts with an inclusive hexadecimal start address
and an exclusive hexadecimal end address. The existing name and size fields
may follow those two addresses.

After capture, run the new postprocessor with the map:

```sh
util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing \
  /path/to/raw-trace-directory -j 4 --weight-map /path/to/weightsAddress.map
```

It produces a normal `kernelslist.g` and `.tracez` files. The `--text` option
still produces `.traceg` files. If `--weight-map` is omitted, processing uses
the upstream behavior. A missing or malformed map causes a nonzero exit.

The marker follows the old fork's behavior: only the retained global-memory
half of an `LDGSTS` pair is marked when its first reported address falls
inside a weight range. The instruction opcode gains `.WEIGHT`; all other
instruction fields and addresses are preserved. Other weight-loading opcode
families are outside this port's current classification rule.

## Simulation

Use a combined A100 configuration with the simple DRAM model enabled and set
both latency options, for example:

```text
-gpgpu_simple_dram_model 1
-dram_latency 100
-dram_hbf_latency 110
```

`-dram_hbf_latency` affects only memory requests propagated from `.WEIGHT`
instructions when they reach the simple DRAM latency queue. Ordinary requests
use `-dram_latency`; the detailed DRAM timing model is unchanged. The old
WeightAddress A100 configurations can be used with this branch.

The Accel-Sim and GPGPU-Sim repositories must both be checked out on their
`hbf-v2.0.1` branches to build and run this port.

## Validation

Validation artifacts are under
`/home/lsc/HBF/nfs_hw_run/lsc/hbf_validation/hbf_port_v201_20261029/`.
The new postprocessor generated 38 `.tracez` files for a fresh Qwen3-MoE
`bs1_seq1` self-attention capture. It marked 131,072 `LDGSTS` loads in each
of the two large GEMM kernels. A text-format check found 2,509,136 lines in
kernel 21 and zero differences from an unmarked conversion after removing
the `.WEIGHT` suffix.

The new simulator completed all 38 marked kernels with the old A100 simple
DRAM configuration (`-dram_latency 100`, `-dram_hbf_latency 110`): exit code
0 and 388,498 total simulated cycles. For kernel 21 alone:

| Trace and configuration | Simulated cycles |
| --- | ---: |
| Unmarked, DRAM 100 / HBF 110 | 83,222 |
| Marked, DRAM 100 / HBF 100 | 83,222 |
| Marked, DRAM 100 / HBF 110 | 82,457 |
| Marked, DRAM 100 / HBF 500 | 176,925 |

The equality at 100 checks that the suffix alone does not alter timing when
both memory paths have equal latency. The 500-cycle setting confirms that
marked requests reach the HBF latency path. The 100-to-110 result is not
monotonic, so it should be treated as simulator scheduling behavior rather
than a calibrated hardware prediction. No physical HBF timing reference was
used for this validation.

As a regression check, the ported simulator replayed the same unmarked kernel
21 with the stock A100 configuration in 167,474 cycles, exactly matching the
pre-port v2.0.1 result for that kernel and configuration.
