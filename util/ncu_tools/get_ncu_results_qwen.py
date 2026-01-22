import re
import argparse
import os

# python get_ncu_results_qwen.py /home/lsc/HBF/nfs_hw_run/lsc/ncu_profiles/device-0/11.7/Qwen3/run_qwen3_decodeLayer_1_50.txt --range 12,69 -p /home/lsc/HBF/accel-sim-framework/result/Qwen3/ncu_results -f output_1_50.csv
def main():
    parser = argparse.ArgumentParser(description="Extract 'Elapsed Cycles' from NCU profile output.")
    parser.add_argument("file_path", help="Path to the NCU output .txt file")
    parser.add_argument("--range", "-r", 
                        help="Kernel index range to keep, e.g., '2,5' (inclusive, 1-based)",
                        type=str, default=None)
    parser.add_argument("--output_path", "-p",
                        help="Directory path to save the output file (default: current directory)",
                        type=str, default=".")
    parser.add_argument("--output_filename", "-f",
                        help="Filename for the output (default: ncu_result.txt)",
                        type=str, default="ncu_result.txt")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, args.output_filename)

    # 读取文件
    try:
        with open(args.file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.file_path}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    elapsed_cycles = []

    # 正则匹配 Elapsed Cycles 行
    pattern = re.compile(r'^\s*Elapsed Cycles\s+cycle\s+([\d,]+)\s*$')

    for line in lines:
        match = pattern.match(line)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                num = int(num_str)
                elapsed_cycles.append(num)
            except ValueError:
                print(f"⚠️ Warning: Failed to parse number: {match.group(1)}")

    total_kernels = len(elapsed_cycles)
    summary_lines = [f"Total kernels found: {total_kernels}"]

    # 处理 range 参数
    selected_cycles = elapsed_cycles
    range_str = "1~{}".format(total_kernels)  # 默认范围
    if args.range:
        try:
            parts = args.range.split(',')
            if len(parts) != 2:
                raise ValueError("Range must have exactly two numbers separated by comma.")
            start_idx = int(parts[0])
            end_idx = int(parts[1])

            if start_idx < 1 or end_idx < start_idx or end_idx > total_kernels:
                raise ValueError(f"Invalid range: indices must be >=1, end >= start, and <= {total_kernels}.")

            selected_cycles = elapsed_cycles[start_idx - 1 : end_idx]
            range_str = f"{start_idx}~{end_idx}"
        except Exception as e:
            error_msg = f"❌ Invalid --range argument: {e}"
            print(error_msg)
            summary_lines.append(error_msg)
            # 即使出错也写入日志
            with open(output_file, 'w') as outf:
                outf.write("\n".join(summary_lines))
            return

    # 构造输出内容
    summary_lines.append(f"kernel_range: {range_str}")
    summary_lines.append(str(selected_cycles))

    # 打印到控制台
    for line in summary_lines:
        print(line)

    # 写入文件
    try:
        with open(output_file, 'w') as outf:
            outf.write("\n".join(summary_lines))
        print(f"\n✅ Output written to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to write output file: {e}")

if __name__ == "__main__":
    main()