#!/usr/bin/env python3
import os
import sys
import datetime
import subprocess
from optparse import OptionParser

# import common from job_launching
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.insert(0, os.path.join(this_directory, "..", "job_launching"))
import common

import tempfile, os, shutil
tmpdir = os.path.join(os.getcwd(), "tmp_ncu")
os.makedirs(tmpdir, exist_ok=True)
os.environ["TMPDIR"] = tmpdir

# -------------------------------
# 参数解析
# -------------------------------
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                  default="rodinia_2.0-ft",
                  help="comma separated list of benchmark suites to run (see apps/define-*.yml)")
parser.add_option("-D", "--device_num", dest="device_num", default="0",
                  help="CUDA device number")
parser.add_option("-n", "--norun", dest="norun", action="store_true",
                  help="Do not actually run the apps, just create run scripts")
parser.add_option("--extra_args", dest="extra_args", default="",
                  help="Additional CLI args for ncu, e.g. '--section MemoryWorkloadAnalysis'")

(options, args) = parser.parse_args()

common.load_defined_yamls()
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))
cuda_version = common.get_cuda_version(this_directory)

# -------------------------------
# 全局路径与日志
# -------------------------------
now_time = datetime.datetime.now()
base_dir = os.path.join(this_directory, "..", "..", "nfs_hw_run", "lsc", "ncu_profiles")

# -------------------------------
# 主循环：遍历 benchmark
# -------------------------------
for bench in benchmarks:
    edir, ddir, exe, argslist = bench

    for argpair in argslist:
        args = argpair["args"] or ""
        run_name = os.path.join(exe, common.get_argfoldername(args))
        this_run_dir = os.path.abspath(
            os.path.expandvars(
                os.path.join(base_dir, f"device-{options.device_num}",
                             cuda_version, run_name)
            )
        )
        os.makedirs(this_run_dir, exist_ok=True)

        # 链接数据目录
        try:
            benchmark_data_dir = common.dir_option_test(os.path.join(ddir, exe, "data"), "", this_directory)
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(benchmark_data_dir, os.path.join(this_run_dir, "data"))
        except common.PathMissing:
            pass

        all_data_link = os.path.join(this_run_dir, "data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        top_data_dir_path = common.dir_option_test(ddir, "", this_directory)
        os.symlink(top_data_dir_path, all_data_link)

        exec_path = common.file_option_test(os.path.join(edir, exe), "", this_directory)

        # -------------------------------
        # 生成 run_ncu.sh
        # -------------------------------
        arg_suffix = "_".join(args.split()) if args else "default"
        txt_out = os.path.join(this_run_dir, f"{exe}_{arg_suffix}.txt")

        sh_contents = f"""#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES={options.device_num}
echo "Running Nsight Compute profiling for {exe} ..."
ncu {options.extra_args} {exec_path} {args} | tee {txt_out}
echo "Profiling completed: results saved in {txt_out}"
"""

        run_sh = os.path.join(this_run_dir, "run_ncu.sh")
        with open(run_sh, "w") as f:
            f.write(sh_contents)
        subprocess.call(["chmod", "u+x", run_sh])

        # -------------------------------
        # 执行
        # -------------------------------
        if not options.norun:
            print(f"\n[Running] {exe}  args={args}")
            os.chdir(this_run_dir)
            ret = subprocess.call(["bash", "run_ncu.sh"])
            if ret != 0:
                sys.exit(f"Error invoking ncu on {this_run_dir}")
            os.chdir(this_directory)

print("\n✅ All profiling runs completed successfully.")
