# Nsight Compute 批量分析脚本说明

## 🧩 文件名
`run_hw_trace.py`

该脚本用于自动化执行 **Nsight Compute (ncu)** 对多个 GPU benchmark 程序的性能分析。它会根据配置批量生成并运行 `run_ncu.sh` 脚本，并将输出结果按 benchmark 分类保存。

---

## 🚀 基本用法

```bash
python3 run_hw_trace.py [options]
```

示例：

```
python3 run_hw_trace.py -B rodinia_2.0-ft -D 0 --extra_args "--section MemoryWorkloadAnalysis"
```

------

## ⚙️ 参数说明

| 参数                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `-B`, `--benchmark_list` | 指定要运行的 benchmark 集合名称（多个用逗号分隔），对应于 `apps/define-*.yml` 文件中定义的测试集合。 默认：`rodinia_2.0-ft` |
| `-D`, `--device_num`     | 指定运行 Nsight Compute 的 CUDA 设备编号（GPU ID）。 默认：`0` |
| `-n`, `--norun`          | 仅生成 `run_ncu.sh` 脚本但**不执行** Nsight Compute。 适合批量预生成配置后手动执行。 |
| `--extra_args`           | 向 `ncu` 传递额外命令行参数，例如： `--extra_args "--section MemoryWorkloadAnalysis"` 或 `--extra_args "--metrics sm__throughput.avg.pct_of_peak_sustained_active"` |

------

## 🧠 工作机制

1. 通过 `common.load_defined_yamls()` 读取并展开定义的 benchmark 列表；
2. 对每个 benchmark：
   - 建立输出目录（形如 `nfs_hw_run/lsc/ncu_profiles/device-0/...`）；
   - 自动创建数据软链接；
   - 生成并保存 `run_ncu.sh`；
   - 执行 `ncu` 命令并将分析结果输出至对应 `.txt` 文件；
3. 若启用 `--norun` 参数，则仅生成脚本文件，不执行分析。

------

## 📁 输出目录结构

生成的结果会保存在类似如下路径：

```
nfs_hw_run/lsc/ncu_profiles/
└── device-0/
    └── 12.3/                  # CUDA 版本
        ├── benchmark1/
        │   ├── run_ncu.sh
        │   ├── benchmark1_default.txt
        │   └── data -> /path/to/data
        └── benchmark2/
```

------

## 📄 输出内容

每个 benchmark 执行后会生成：

- `run_ncu.sh` — Nsight Compute 执行脚本；
- `xxx_args.txt` — Nsight Compute 的分析输出（包含性能指标和摘要信息）。

------

## ✅ 示例

```
# 仅生成执行脚本
python3 run_hw_trace.py -B rodinia_2.0-ft -D 1 -n

# 实际运行 profiling，并输出 MemoryWorkloadAnalysis section
python3 run_hw_trace.py -B rodinia_2.0-ft -D 0 --extra_args "--section MemoryWorkloadAnalysis"
```

执行完成后，控制台输出：

```
[Running] hotSpot  args=
Running Nsight Compute profiling for hotSpot ...
Profiling completed: results saved in hotSpot_default.txt

✅ All profiling runs completed successfully.
```