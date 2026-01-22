# python get_results_qwen.py bs64_seq1
import os
import re
import argparse
import glob
import csv

def process_single_run(run_dir):
    """
    Process one run directory (e.g., bs1_seq1/) that contains A100_* subdirs.
    Returns True if successful, False otherwise.
    """
    run_name = os.path.basename(os.path.abspath(run_dir))
    print(f"\n{'='*70}")
    print(f"Processing run: {run_name}")
    print(f"{'='*70}")

    pattern = re.compile(r'gpu_sim_cycle\s*=\s*(\d+)')

    # Step 1: Extract gpu_sim_cycle from all gpgpu-sim-out_*.txt files
    config_cycles = {}
    for dir_name in os.listdir(run_dir):
        dir_path = os.path.join(run_dir, dir_name)
        if os.path.isdir(dir_path) and dir_name.startswith("A100"):
            print(f"  Processing config: {dir_name}")
            cycles_list = []
            for file_name in os.listdir(dir_path):
                if file_name.startswith("gpgpu-sim-out_") and file_name.endswith(".txt"):
                    file_path = os.path.join(dir_path, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            matches = pattern.findall(content)
                            cycles_list.extend(int(m) for m in matches)
                    except Exception as e:
                        print(f"    ⚠️ Error reading {file_path}: {e}")
            config_cycles[dir_name] = cycles_list
            print(f"    → Found {len(cycles_list)} cycles")

    if not config_cycles:
        print(f"  ❌ No A100 configurations found in {run_dir}. Skipping.")
        return False

    # Step 2: Extract kernel names from A100-SASS/traces/
    sass_dir = os.path.join(run_dir, "A100-SASS")
    traces_dir = os.path.join(sass_dir, "traces")
    kernel_names = []

    kernelslist_path = os.path.join(traces_dir, "kernelslist.g")
    if not os.path.isfile(kernelslist_path):
        print(f"  ⚠️ kernelslist.g not found at {kernelslist_path}")
    else:
        simulated_kernels = []
        try:
            with open(kernelslist_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.endswith('.traceg.xz'):
                        base_name = line[:-len('.traceg.xz')]
                        simulated_kernels.append(base_name)
            print(f"  Loaded {len(simulated_kernels)} kernels from kernelslist.g")
        except Exception as e:
            print(f"  ⚠️ Error reading kernelslist.g: {e}")
            simulated_kernels = []

        # Find stats_ctx_* file
        stats_files = glob.glob(os.path.join(traces_dir, "stats_ctx_*"))
        stats_files = [f for f in stats_files if os.path.isfile(f)]

        if len(stats_files) != 1:
            print(f"  ⚠️ Expected exactly one stats_ctx_* file in {traces_dir}, found {len(stats_files)}")
        else:
            stats_file = stats_files[0]
            print(f"  Reading kernel metadata from: {stats_file}")

            trace_to_kernel_name = {}
            try:
                with open(stats_file, 'r', encoding='utf-8', errors='ignore') as f:
                    next(f)  # skip header
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(',', 1)
                        if len(parts) < 2:
                            continue
                        trace_filename = parts[0].strip()
                        kernel_name = parts[1].split(',')[0].strip()

                        base = trace_filename
                        if base.endswith('.trace.xz'):
                            base = base[:-len('.trace.xz')]
                        elif base.endswith('.traceg.xz'):
                            base = base[:-len('.traceg.xz')]
                        elif base.endswith('.xz'):
                            base = base.rsplit('.', 1)[0]

                        trace_to_kernel_name[base] = kernel_name
            except Exception as e:
                print(f"  ⚠️ Error parsing {stats_file}: {e}")

            for base in simulated_kernels:
                kernel_names.append(trace_to_kernel_name.get(base, "<UNKNOWN>"))

    n_kernels = len(kernel_names)
    print(f"  Total kernels extracted: {n_kernels}")

    # Step 3: Validate and align configs
    config_order = [
        "A100-SASS",
        "A100_simpleDram_Latency160_100-SASS",
        "A100_simpleDram_Latency160_150-SASS",
        "A100_simpleDram_Latency160_200-SASS",
        "A100_simpleDram_Latency160_300-SASS",
        "A100_simpleDram_Latency160_400-SASS",
        "A100_simpleDram_Latency160_500-SASS",
        "A100_simpleDram_Latency160_1000-SASS"
    ]

    valid_configs = []
    for cfg in config_order:
        if cfg not in config_cycles:
            continue
        cycles = config_cycles[cfg]
        if len(cycles) != n_kernels:
            print(f"  ⚠️ Skipping {cfg}: cycle count ({len(cycles)}) ≠ kernel count ({n_kernels})")
        else:
            valid_configs.append(cfg)

    if not valid_configs or n_kernels == 0:
        print(f"  ❌ Not enough data to generate CSV for {run_name}.")
        return False

    # Step 4: Write CSV
    output_csv = f"output_{run_name}.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Kernel Name"] + valid_configs)
        for i in range(n_kernels):
            row = [kernel_names[i]] + [config_cycles[cfg][i] for cfg in valid_configs]
            writer.writerow(row)

    print(f"  ✅ CSV written: {output_csv}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract gpu_sim_cycle from two-level GPGPU-Sim outputs.")
    parser.add_argument("root_dir", help="Root directory containing bsX_seqY/ subdirectories")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    # Find all second-level run directories (e.g., bs1_seq1, bs2_seq2)
    run_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Optional: filter by pattern like "bs*_seq*"
            if re.match(r'bs\d+_seq\d+', item):
                run_dirs.append(item_path)
            else:
                # Or include all subdirs that contain A100-* folders
                has_a100 = any(name.startswith("A100") for name in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, name)))
                if has_a100:
                    run_dirs.append(item_path)

    if not run_dirs:
        print(f"❌ No valid run directories found in {root_dir}")
        return

    print(f"Found {len(run_dirs)} run(s) to process:")
    for d in run_dirs:
        print(f"  - {os.path.basename(d)}")

    success_count = 0
    for run_dir in sorted(run_dirs):
        if process_single_run(run_dir):
            success_count += 1

    print(f"\n{'='*70}")
    print(f"✅ Completed: {success_count}/{len(run_dirs)} runs processed successfully.")
    print(f"Output CSVs are in current working directory.")


if __name__ == "__main__":
    main()


# import os
# import re
# import argparse
# import glob
# import csv

# # 统计accel_sim的输出结果
# # python get_results_qwen.py 1_1
# def main():
#     parser = argparse.ArgumentParser(description="Extract gpu_sim_cycle and kernel names from GPGPU-Sim outputs.")
#     parser.add_argument("base_dir", help="Path to the root directory containing A100_* subdirectories")
#     args = parser.parse_args()

#     base_dir = os.path.abspath(args.base_dir)
#     pattern = re.compile(r'gpu_sim_cycle\s*=\s*(\d+)')

#     if not os.path.isdir(base_dir):
#         print(f"Error: '{base_dir}' is not a valid directory.")
#         return

#     # Step 1: Extract gpu_sim_cycle from all gpgpu-sim-out_*.txt files
#     config_cycles = {}
#     for dir_name in os.listdir(base_dir):
#         dir_path = os.path.join(base_dir, dir_name)
#         if os.path.isdir(dir_path) and dir_name.startswith("A100"):
#             print(f"Processing directory: {dir_name}")
#             cycles_list = []
#             for file_name in os.listdir(dir_path):
#                 if file_name.startswith("gpgpu-sim-out_") and file_name.endswith(".txt"):
#                     file_path = os.path.join(dir_path, file_name)
#                     try:
#                         with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                             content = f.read()
#                             matches = pattern.findall(content)
#                             cycles_list.extend(int(m) for m in matches)
#                     except Exception as e:
#                         print(f"  ⚠️ Error reading {file_path}: {e}")
#             config_cycles[dir_name] = cycles_list
#             print(f"  → Found {len(cycles_list)} cycles")

#     # Step 2: Extract kernel names in order from kernelslist.g + stats_ctx_*
#     sass_dir = os.path.join(base_dir, "A100-SASS")
#     traces_dir = os.path.join(sass_dir, "traces")
#     kernel_names = []

#     kernelslist_path = os.path.join(traces_dir, "kernelslist.g")
#     if not os.path.isfile(kernelslist_path):
#         print(f"⚠️ kernelslist.g not found at {kernelslist_path}")
#     else:
#         simulated_kernels = []
#         try:
#             with open(kernelslist_path, 'r') as f:
#                 for line in f:
#                     line = line.strip()
#                     if line and line.endswith('.traceg.xz'):
#                         base_name = line[:-len('.traceg.xz')]
#                         simulated_kernels.append(base_name)
#             print(f"\nLoaded {len(simulated_kernels)} kernels from kernelslist.g")
#         except Exception as e:
#             print(f"⚠️ Error reading kernelslist.g: {e}")
#             simulated_kernels = []

#         # Find stats_ctx_* file
#         stats_files = glob.glob(os.path.join(traces_dir, "stats_ctx_*"))
#         stats_files = [f for f in stats_files if os.path.isfile(f)]

#         if len(stats_files) != 1:
#             print(f"⚠️ Expected exactly one stats_ctx_* file in {traces_dir}, found {len(stats_files)}")
#         else:
#             stats_file = stats_files[0]
#             print(f"Reading kernel metadata from: {stats_file}")

#             trace_to_kernel_name = {}
#             try:
#                 with open(stats_file, 'r', encoding='utf-8', errors='ignore') as f:
#                     next(f)  # skip header
#                     for line in f:
#                         line = line.strip()
#                         if not line:
#                             continue
#                         parts = line.split(',', 1)
#                         if len(parts) < 2:
#                             continue
#                         trace_filename = parts[0].strip()
#                         kernel_name = parts[1].split(',')[0].strip()

#                         # Normalize base name (remove .trace.xz / .traceg.xz)
#                         base = trace_filename
#                         if base.endswith('.trace.xz'):
#                             base = base[:-len('.trace.xz')]
#                         elif base.endswith('.traceg.xz'):
#                             base = base[:-len('.traceg.xz')]
#                         elif base.endswith('.xz'):
#                             base = base.rsplit('.', 1)[0]

#                         trace_to_kernel_name[base] = kernel_name
#             except Exception as e:
#                 print(f"⚠️ Error parsing {stats_file}: {e}")

#             for base in simulated_kernels:
#                 kernel_names.append(trace_to_kernel_name.get(base, "<UNKNOWN>"))

#     n_kernels = len(kernel_names)
#     print(f"\nTotal kernels extracted: {n_kernels}")

#     # Step 3: Print summary
#     print("\n" + "="*60)
#     print("Final Results: One array per A100 configuration")
#     print("="*60)
#     for config_name, cycles in config_cycles.items():
#         print(f"\n{config_name}:")
#         print(f"  Cycles: {cycles}")
#         print(f"  Count: {len(cycles)}")

#     print("\n" + "="*60)
#     print("Kernel Names (from A100-SASS/traces/stats_ctx_*.txt)")
#     print("="*60)
#     for i, name in enumerate(kernel_names, start=1):
#         print(f"{i:2d}: {name}")

#     # Step 4: Output to CSV — auto-align by kernel index
#     if n_kernels == 0:
#         print("\n❌ No kernel names found. Skipping CSV output.")
#         return

#     # Define desired column order (put A100-SASS first, then others sorted)
#     desired_order = ["A100-SASS"]
#     other_configs = sorted([cfg for cfg in config_cycles.keys() if cfg != "A100-SASS"])
#     config_order = [
#         "A100-SASS",
#         "A100_simpleDram_Latency160_100-SASS",
#         "A100_simpleDram_Latency160_150-SASS",
#         "A100_simpleDram_Latency160_200-SASS",
#         "A100_simpleDram_Latency160_300-SASS",
#         "A100_simpleDram_Latency160_400-SASS",
#         "A100_simpleDram_Latency160_500-SASS",
#         "A100_simpleDram_Latency160_1000-SASS"
#     ]

#     # Validate all configs have same cycle count as kernel_names
#     valid_configs = []
#     for cfg in config_order:
#         if cfg not in config_cycles:
#             continue
#         cycles = config_cycles[cfg]
#         if len(cycles) != n_kernels:
#             print(f"⚠️ Skipping {cfg}: cycle count ({len(cycles)}) ≠ kernel count ({n_kernels})")
#         else:
#             valid_configs.append(cfg)

#     if not valid_configs:
#         print("\n❌ No valid configurations to output.")
#         return

#     # Write CSV
#     output_csv = f"output_{os.path.basename(base_dir.rstrip('/'))}.csv"
#     with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Kernel Name"] + valid_configs)
#         for i in range(n_kernels):
#             row = [kernel_names[i]] + [config_cycles[cfg][i] for cfg in valid_configs]
#             writer.writerow(row)

#     print(f"\n✅ CSV output written to: {output_csv}")

# if __name__ == "__main__":
#     main()
