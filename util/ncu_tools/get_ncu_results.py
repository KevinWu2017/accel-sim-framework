#!/usr/bin/env python3
import os
import sys
import re
import csv
import datetime
from optparse import OptionParser

# import common from job_launching
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.insert(0, os.path.join(this_directory, "..", "job_launching"))
import common

# -------------------------------
# 参数解析
# -------------------------------
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                  default="rodinia_2.0-ft",
                  help="comma separated list of benchmark suites to read (see apps/define-*.yml)")
parser.add_option("-D", "--device_num", dest="device_num", default="0",
                  help="CUDA device number")
parser.add_option("--metric", dest="metric", default="Elapsed Cycles",
                  help="Metric name to extract, default: 'Elapsed Cycles'")

(options, args) = parser.parse_args()

# -------------------------------
# 初始化环境
# -------------------------------
common.load_defined_yamls()
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))
cuda_version = common.get_cuda_version(this_directory)

base_dir = os.path.join(this_directory, "..", "..", "nfs_hw_run", "lsc", "ncu_profiles")
output_csv = os.path.join(
    base_dir,
    f"ncu_{'_'.join(options.benchmark_list.split(','))}.csv"
)
# -------------------------------
# 正则表达式匹配
# -------------------------------
metric_pattern = re.compile(
    rf"^\s*{re.escape(options.metric)}\s+\S+\s+([\d,\.]+)", re.MULTILINE
)


results = []

# -------------------------------
# 遍历所有 benchmark
# -------------------------------
for bench in benchmarks:
    edir, ddir, exe, argslist = bench

    for argpair in argslist:
        args = argpair["args"] or ""
        run_name = os.path.join(exe, common.get_argfoldername(args))
        run_dir = os.path.abspath(
            os.path.expandvars(
                os.path.join(base_dir, f"device-{options.device_num}",
                             cuda_version, run_name)
            )
        )
        if not os.path.exists(run_dir):
            print(f"⚠️ Skip: {run_dir} not found")
            continue

        # 查找 run_ncu.sh 生成的 txt 输出
        txt_files = [f for f in os.listdir(run_dir) if f.endswith(".txt")]
        if not txt_files:
            print(f"⚠️ No .txt result in {run_dir}")
            continue

        txt_path = os.path.join(run_dir, txt_files[0])
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # 匹配 metric
        match = metric_pattern.search(content)
        if match:
            value_str = match.group(1).replace(",", "")
            try:
                value = float(value_str)
            except ValueError:
                value = None
        else:
            value = None

        results.append({
            "benchmark": exe,
            "args": args,
            "metric_name": options.metric,
            "metric_value": value,
            "file": txt_files[0]
        })

# -------------------------------
# 输出 CSV
# -------------------------------
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["benchmark", "args", "metric_name", "metric_value", "file"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Done. Extracted {len(results)} results into:\n{output_csv}")
