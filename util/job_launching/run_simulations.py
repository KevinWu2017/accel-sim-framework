#!/usr/bin/env python3

# Copyright (c) 2018-2021, Mahmoud Khairy, Vijay Kandiah, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
# Northwestern University, Purdue University, The University of British Columbia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer;
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution;
# 3. Neither the names of Northwestern University, Purdue University,
#    The University of British Columbia nor the names of their contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from optparse import OptionParser
import os
import subprocess
from subprocess import Popen, PIPE

import sys
import re
import shutil
import glob
import datetime
import yaml
import common

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
# This function will pull the SO name out of the shared object,
# which will have current GIT commit number attatched.
def extract_version(exec_path, simulator):
    if simulator == "gpgpusim":
        regex_base = "gpgpu-sim_git-commit"
    else:
        regex_base = "accelsim-commit"
    regex_str = r".*({0}[^\s^']+).*".format(regex_base)
    strings_process = Popen(["strings", exec_path], stdout=PIPE)
    grep_process = Popen(
        ["grep", regex_base], stdin=strings_process.stdout, stdout=PIPE
    )
    strings_process.stdout.close()
    out, err = grep_process.communicate()
    version = re.sub(regex_str, r"\1", str(out.rstrip()))
    return version


#######################################################################################
# Class the represents each configuration you are going to run
# For example, if your sweep file has 2 entries 32k-L1 and 64k-L1 there will be 2
# ConfigurationSpec classes and the run_subdir name for each will be 32k-L1 and 64k-L1
# respectively
class ConfigurationSpec:
    #########################################################################################
    # Public Interface methods
    #########################################################################################
    # Class is constructed with a single line of text from the sweep_param file
    def __init__(self, nameTuple):
        name, params, config_file = nameTuple
        self.run_subdir = name
        self.params = params
        self.config_file = config_file

    def my_print(self):
        print("Run Subdir = " + self.run_subdir)
        print("Parameters = " + self.params)
        print("Base config file = " + self.config_file)

    # 为一组基准测试程序（benchmarks）生成配置文件、设置运行目录，并通过 Torque 资源管理器提交模拟任务到集群中执行。
    def run(self, build_handle, benchmarks, run_directory, cuda_version, simdir):
        # 对每一个基准测试进行处理，解包出可执行路径、数据路径、名称以及参数列表。
        for dir_bench in benchmarks:
            exec_dir, data_dir, benchmark, self.command_line_args_list = dir_bench
            # 设置执行和数据目录  Trace模式不需要
            full_exec_dir = ""  # For traces it is not necessary to have the apps built
            full_data_dir = ""
            if options.trace_dir == "":
                full_exec_dir = common.dir_option_test(
                    os.path.expandvars(exec_dir), "", this_directory
                )
                try:
                    full_data_dir = common.dir_option_test(
                        os.path.join(
                            os.path.expandvars(data_dir), benchmark.replace("/", "_")
                        ),
                        "",
                        this_directory,
                    )
                except common.PathMissing:
                    pass

            # 为每组命令行参数创建子目录名
            self.benchmark_args_subdirs = {}
            for argmap in self.command_line_args_list:
                args = argmap["args"]
                self.benchmark_args_subdirs[args] = common.get_argfoldername(args)

            # 再次循环：为每组参数准备和提交作业
            for argmap in self.command_line_args_list:
                args = argmap["args"]
                mem_usage = argmap["accel-sim-mem"]
                appargs_run_subdir = os.path.join(
                    benchmark.replace("/", "_"), self.benchmark_args_subdirs[args]
                )
                this_run_dir = os.path.join(
                    run_directory, appargs_run_subdir, self.run_subdir
                )
                #  创建运行目录并复制必要资源
                self.setup_run_directory(
                    full_data_dir, this_run_dir, data_dir, appargs_run_subdir
                )

                # 生成 Torque 提交脚本（PBS）
                self.text_replace_torque_sim(
                    full_data_dir,
                    this_run_dir,
                    benchmark,
                    cuda_version,
                    args,
                    simdir,
                    full_exec_dir,
                    build_handle,
                    mem_usage,
                )
                # 生成 GPGPU-Sim 配置文件
                self.append_gpgpusim_config(
                    benchmark, this_run_dir, appargs_run_subdir, self.config_file
                )

                # 提交作业到 Torque 队列系统
                # Submit the job to torque and dump the output to a file
                if not options.no_launch:
                    torque_out_filename = this_directory + "torque_out.{0}.txt".format(
                        os.getpid()
                    )
                    torque_out_file = open(torque_out_filename, "w+")
                    saved_dir = os.getcwd()
                    os.chdir(this_run_dir)
                    if (
                        subprocess.call(
                            [job_submit_call, os.path.join(this_run_dir, job_template)],
                            stdout=torque_out_file,
                        )
                        < 0
                    ):
                        exit("Error Launching Job")
                    else:
                        # Parse the torque output for just the numeric ID
                        torque_out_file.seek(0)
                        torque_out = re.sub(
                            r"[^\d]*(\d*).*", r"\1", torque_out_file.read().strip()
                        )
                        print(
                            "Job "
                            + torque_out
                            + " queued ("
                            + benchmark
                            + "-"
                            + self.benchmark_args_subdirs[args]
                            + " "
                            + self.run_subdir
                            + ")"
                        )
                    torque_out_file.close()
                    os.remove(torque_out_filename)
                    os.chdir(saved_dir)

                    # 记录日志信息
                    if len(torque_out) > 0:
                        # Dump the benchmark description to the logfile
                        if not os.path.exists(this_directory + "logfiles/"):
                            # In the very rare case that concurrent builds try to make the directory at the same time
                            # (after the test to os.path.exists -- this has actually happened...)
                            try:
                                os.makedirs(this_directory + "logfiles/")
                            except:
                                pass
                        now_time = datetime.datetime.now()
                        day_string = now_time.strftime("%y.%m.%d-%A")
                        time_string = now_time.strftime("%H:%M:%S")
                        log_name = "sim_log.{0}".format(options.launch_name)
                        logfile = open(
                            this_directory
                            + "logfiles/"
                            + log_name
                            + "."
                            + day_string
                            + ".txt",
                            "a",
                        )
                        print(
                            "%s %6s %-22s %-100s %-25s %s"
                            % (
                                time_string,
                                torque_out,
                                benchmark,
                                self.benchmark_args_subdirs[args],
                                self.run_subdir,
                                build_handle,
                            ),
                            file=logfile,
                        )
                        logfile.close()
            self.benchmark_args_subdirs.clear()

    #########################################################################################
    # Internal utility methods
    #########################################################################################
    # copies and links the necessary files to the run directory
    def setup_run_directory(
        self, full_data_dir, this_run_dir, data_dir, appargs_subdir
    ):
        if not os.path.isdir(this_run_dir):
            os.makedirs(this_run_dir)

        files_to_copy_to_run_dir = (
            glob.glob(os.path.join(full_data_dir, "*.ptx"))
            + glob.glob(os.path.join(full_data_dir, "*.cl"))
            + glob.glob(os.path.join(full_data_dir, "*.h"))
            + glob.glob(os.path.dirname(self.config_file) + "/*.icnt")
            + glob.glob(os.path.dirname(self.config_file) + "/*.csv")
            + glob.glob(os.path.dirname(self.config_file) + "/*.xml")
        )

        for file_to_cp in files_to_copy_to_run_dir:
            new_file = os.path.join(
                this_run_dir, os.path.basename(this_directory + file_to_cp)
            )
            if os.path.isfile(new_file):
                os.remove(new_file)
            shutil.copyfile(file_to_cp, new_file)

        # link the data directory
        benchmark_data_dir = os.path.join(full_data_dir, "data")
        if os.path.isdir(benchmark_data_dir):
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(benchmark_data_dir, os.path.join(this_run_dir, "data"))

        # link the traces directory
        if options.trace_dir != "":

            ### This code handles the case where you pass a directory a few levels up from the
            ### directory where the traces are laid out.
            benchmark_trace_dir = None
            paths_to_try = (
                [os.path.join(options.trace_dir, appargs_subdir, "traces")]
                + glob.glob(
                    os.path.join(
                        options.trace_dir, "**", "**", appargs_subdir, "traces"
                    )
                )
                + glob.glob(
                    os.path.join(options.trace_dir, "**", appargs_subdir, "traces")
                )
            )
            for path in paths_to_try:
                try:
                    benchmark_trace_dir = common.dir_option_test(
                        path, "", this_directory
                    )
                    break
                except common.PathMissing as e:
                    pass

            if benchmark_trace_dir == None:
                sys.exit(
                    "Cannot find traces in any of the paths: {0}".format(paths_to_try)
                )
            benchmark_trace_dir = os.path.abspath(benchmark_trace_dir)
            if os.path.isdir(benchmark_trace_dir):
                if os.path.lexists(os.path.join(this_run_dir, "traces")):
                    os.remove(os.path.join(this_run_dir, "traces"))
                os.symlink(benchmark_trace_dir, os.path.join(this_run_dir, "traces"))

        all_data_link = os.path.join(this_run_dir, "data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        if os.path.exists(os.path.join(this_directory, data_dir)):
            os.symlink(os.path.join(this_directory, data_dir), all_data_link)

    # replaces all the "REAPLCE_*" strings in the .sim file
    def text_replace_torque_sim(
        self,
        full_run_dir,
        this_run_dir,
        benchmark,
        cuda_version,
        command_line_args,
        libpath,
        exec_dir,
        gpgpusim_build_handle,
        mem_usage,
    ):
        # get the pre-launch sh commands
        prelaunch_filename = full_run_dir + "benchmark_pre_launch_command_line.txt"
        benchmark_command_line = ""
        if os.path.isfile(prelaunch_filename):
            f = open(prelaunch_filename)
            benchmark_command_line = f.read().strip()
            f.close()

        if options.trace_dir == "":
            # If the config contains "SASS" and you have not specified the trace directory, then likely something is wrong
            if ("SASS" in self.run_subdir):
                print("You are trying to run a configuration with SASS in it, but have not specified a trace directory."+\
                      " If you want to run SASS traces, please specify -T to point to the top-level trace directory")
                exit(1)
            exec_name = (
                options.benchmark_exec_prefix
                + " "
                + os.path.join(this_directory, exec_dir, benchmark)
            )
        else:
            exec_name = (
                options.benchmark_exec_prefix
                + " "
                + os.path.join(libpath, "accel-sim.out")
            )

        # Test the existance of required env variables
        if str(os.getenv("GPGPUSIM_ROOT")) == "None":
            exit("\nERROR - Specify GPGPUSIM_ROOT prior to running this script")
        if str(os.getenv("OPENCL_REMOTE_GPU_HOST")) == "None":
            os.environ["OPENCL_REMOTE_GPU_HOST"] = ""
        if str(os.getenv("GPGPUSIM_CONFIG")) == "None":
            exit("\nERROR - Specify GPGPUSIM_CONFIG prior to running this script")

        # If the user specified the memory use that
        if options.job_mem != None:
            mem_usage = options.job_mem
        # if we are using PTX (GPGPU-Sim) - then just assume 4G
        elif options.trace_dir == "":
            mem_usage = "4G"
        if options.trace_dir == "":
            if command_line_args == None:
                txt_args = ""
            else:
                txt_args = str(command_line_args)
        else:
            txt_args = " -config ./gpgpusim.config -trace ./traces/kernelslist.g"

        if os.getenv("TORQUE_QUEUE_NAME") == None:
            queue_name = "batch"
        else:
            queue_name = os.getenv("TORQUE_QUEUE_NAME")

        # do the text replacement for the .sim file
        sim_name = benchmark + "-" + self.benchmark_args_subdirs[command_line_args] + "." +\
                                gpgpusim_build_handle
        # Truncate long simulation file names
        sim_name = sim_name[:200]
        replacement_dict = {"NAME":sim_name,
                            "NODES":"1", 
                            "GPGPUSIM_ROOT":os.getenv("GPGPUSIM_ROOT"),
                            "LIBPATH": libpath,
                            "SUBDIR":this_run_dir,
                            "OPENCL_REMOTE_GPU_HOST":os.getenv("OPENCL_REMOTE_GPU_HOST"),
                            "BENCHMARK_SPECIFIC_COMMAND":benchmark_command_line,
                            "PATH":os.getenv("PATH"),
                            "EXEC_NAME":exec_name,
                            "QUEUE_NAME":queue_name,
                            "COMMAND_LINE":txt_args,
                            "MEM_USAGE": mem_usage
                            }
        torque_text = open(this_directory + job_template).read().strip()
        for entry in replacement_dict:
            torque_text = re.sub(
                "REPLACE_" + entry, str(replacement_dict[entry]), torque_text
            )
        open(os.path.join(this_run_dir, job_template), "w").write(torque_text)
        exec_line = torque_text.splitlines()[-1]
        justrunfile = os.path.join(this_run_dir , "justrun.sh")
        # open(justrunfile, 'w').write(exec_name + " " + txt_args + "\n")
        open(justrunfile, 'w').write(exec_name + " " + txt_args + " | tee gpgpu-sim-out_`date '+%b_%d_%H:%M.%S'`.txt")
        os.chmod(justrunfile, 0o744)

    # replaces all the "REPLACE_*" strings in the gpgpusim.config file
    def append_gpgpusim_config(
        self, bench_name, this_run_dir, appargs_run_subdir, config_text_file
    ):
        benchmark_spec_opts_file = os.path.expandvars(
            os.path.join(
                "$GPUAPPS_ROOT",
                "benchmarks",
                "app-specific-gpgpu-sim-options",
                bench_name,
                "benchmark_options.txt",
            )
        )
        benchmark_spec_opts = ""
        if os.path.isfile(benchmark_spec_opts_file):
            f = open(benchmark_spec_opts_file)
            benchmark_spec_opts = f.read().strip()
            f.close()

        config_text = open(config_text_file).read()
        config_text += "\n" + benchmark_spec_opts + "\n" + self.params + "\n"

        if options.accelwattch_HW:
            # if bench_name == "cutlass_perf_test":
            #     config_text += "\n" + "-hw_perf_bench_name " + appargs_run_subdir.replace('/','_') + "\n"
            # else:
            config_text += "\n" + "-hw_perf_bench_name " + bench_name + "\n"

        if options.trace_dir != "":
            cfgsubdir = re.sub(r".*(configs.*)gpgpusim.config", r"\1", config_text_file)
            config_text += "\n" + "# Accel-Sim Parameters" + "\n"
            accelsim_cfg = os.path.expandvars(
                os.path.join("$ACCELSIM_ROOT", cfgsubdir, "trace.config")
            )
            config_text += open(accelsim_cfg).read()

        open(os.path.join(this_run_dir, "gpgpusim.config"), "w").write(config_text)


# -----------------------------------------------------------
# main script start
# -----------------------------------------------------------
(options, args) = common.parse_run_simulations_options()

if str(os.getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN")) != "1":
    sys.exit("ERROR - Please run setup_environment before running this script")


cuda_version = common.get_cuda_version(this_directory)

# 确定运行目录 默认为根目录下sim_run_11.7
if options.run_directory == "":
    options.run_directory = os.path.join(
        this_directory, "../../sim_run_%s" % cuda_version
    )
else:
    options.run_directory = os.path.join(os.getcwd(), options.run_directory)

#  安全地复制模拟器的可执行文件或动态库（.so 或 accel-sim.out）到一个独立的运行目录中，以避免在运行测试时被正在进行的编译过程覆盖或干扰。
# Let's copy out the .so file so that builds don't interfere with running tests
# If the user does not specify a so file, then use the one in the git repo and copy it out.
# 非 trace 模式：使用 GPGPU-Sim
if options.trace_dir == "":
    options.simulator_dir = common.dir_option_test(
        options.simulator_dir,
        os.path.join(os.getenv("GPGPUSIM_ROOT"), "lib", os.getenv("GPGPUSIM_CONFIG")),
        this_directory,
    )
    # /home/lsc/HBF/gpgpu-sim_distribution/lib/gcc-9.4.0/cuda-11070/release
    simulator_path = os.path.join(options.simulator_dir, "libcudart.so")
 # trace 模式：使用 Accel-Sim 
else:
    options.simulator_dir = common.dir_option_test(
        options.simulator_dir,
        os.path.join(os.getenv("ACCELSIM_ROOT"), "bin", os.getenv("ACCELSIM_CONFIG")),
        this_directory,
    )
    # /home/lsc/HBF/accel-sim-framework/gpu-simulator/bin/release/accel-sim.out
    simulator_path = os.path.join(options.simulator_dir, "accel-sim.out")

# 确定使用模拟器的版本
if options.trace_dir == "":
    # gpgpu-sim_git-commit-b18ee397_modified_0.0
    version_string = extract_version(simulator_path, "gpgpusim")
else:
    gpgpusim_path = os.path.join(
        os.getenv("GPGPUSIM_ROOT"), "lib", os.getenv("GPGPUSIM_CONFIG"), "libcudart.so"
    )
    # accelsim-commit-ff9a5d6_modified_3.0_25-10-24-11-30-33gpgpu-sim_git-commit-b18ee397_modified_0.0
    version_string = extract_version(simulator_path, "accelsim") + extract_version(
        gpgpusim_path, "gpgpusim"
    )
# /sim_run_11.7/gpgpu-sim-builds/{version_string}
running_sim_dir = os.path.join(
    options.run_directory, "gpgpu-sim-builds", version_string
)
if not os.path.exists(running_sim_dir):
    # In the very rare case that concurrent builds try to make the directory at the same time
    # (after the test to os.path.exists -- this has actually happened...)
    try:
        os.makedirs(running_sim_dir)
    except:
        pass

# 安全地将模拟器的可执行文件（或动态库）复制到一个隔离的运行目录中，并更新配置，使后续流程使用这个副本，而不是原始构建目录中的文件。
if not os.path.exists(os.path.join(running_sim_dir, os.path.basename(simulator_path))):
    # 把前者copy到后者中
    shutil.copy(simulator_path, running_sim_dir)
options.simulator_dir = running_sim_dir

common.load_defined_yamls()

# 自动检测系统中可用的作业调度系统（如 Slurm、Torque/PBS）或回退到本地进程管理，并同时 验证 CUDA 编译器 nvcc 是否可用。
# Test for the existance of a cluster management system
job_submit_call = None
job_template = None
if options.launcher != "":
    if options.launcher == "qsub":
        job_submit_call = options.launcher
        job_template = "torque.sim"
    elif options.launcher == "sbatch":
        job_submit_call = options.launcher
        job_template = "slurm.sim"
    elif options.launcher == "local":
        job_submit_call = os.path.join(this_directory, "procman.py")
        job_template = "slurm.sim"
elif any(
    [
        os.path.isfile(os.path.join(p, "sbatch"))
        for p in os.getenv("PATH").split(os.pathsep)
    ]
):
    job_submit_call = "sbatch"
    job_template = "slurm.sim"
elif any(
    [
        os.path.isfile(os.path.join(p, "qsub"))
        for p in os.getenv("PATH").split(os.pathsep)
    ]
):
    job_submit_call = "qsub"
    job_template = "torque.sim"
else:
    print("Cannot find a supported job management system. Spawning jobs locally.")
    job_submit_call = os.path.join(this_directory, "procman.py")
    job_template = "slurm.sim"

if not any(
    [
        os.path.isfile(os.path.join(p, "nvcc"))
        for p in os.getenv("PATH").split(os.pathsep)
    ]
):
    exit(
        "ERROR - Cannot find nvcc PATH... Is CUDA_INSTALL_PATH/bin in the system PATH?"
    )

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cfgs = common.gen_configs_from_list(options.configs_list.split(","))
configurations = []
for config in cfgs:
    configurations.append(ConfigurationSpec(config))

print(
    "Running Simulations with GPGPU-Sim built from \n{0}\n ".format(version_string)
    + "\nUsing configs: "
    + options.configs_list
    + "\nBenchmark: "
    + options.benchmark_list
)

for config in configurations:
    config.my_print()
    config.run(
        version_string,
        benchmarks,
        options.run_directory,
        cuda_version,
        options.simulator_dir,
    )

if "procman" in job_submit_call and not options.no_launch:
    if options.cores == None:
        subprocess.call([job_submit_call, "-S"])
    else:
        subprocess.call([job_submit_call, "-S", "-c", options.cores])
