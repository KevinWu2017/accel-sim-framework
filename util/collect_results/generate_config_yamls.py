import os
import csv
import glob
import argparse
from collections import defaultdict

# python generate_config_yamls.py --input_dir ./visual_final_results/bs64_1 --output_dir ./visual_final_results/bs64_1/configs
# 文件名后缀到 YAML key 的映射
FILENAME_TO_KEY = {
    "inputLayerNorm": "atten_norm",
    "selfAttn": "atten",
    "moeRouter": "moe_route",
    "moeExperts": "moe_experts",
    "postLayerNorm": "moe_norm",
}

SUFFIXES = set(FILENAME_TO_KEY.keys())

def extract_run_id_and_suffix(filename):
    """
    从 time_us_output_XXX_YYY.csv 中提取 run_id 和 suffix。
    要求 YYY 必须是已知的 SUFFIXES 之一。
    """
    if not filename.startswith("time_us_output_") or not filename.endswith(".csv"):
        return None, None

    # 移除前缀和后缀
    stem = filename[len("time_us_output_"):-len(".csv")]
    
    # 尝试每个已知 suffix
    for suffix in SUFFIXES:
        if stem.endswith("_" + suffix):
            run_id = stem[:-(len(suffix) + 1)]  # 去掉 _suffix
            return run_id, suffix

    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing time_us_output_*_*.csv files")
    parser.add_argument("--output_dir", default="./configs", help="Output directory for YAML files")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    time_files = glob.glob(os.path.join(input_dir, "time_us_output_*_*_*.csv"))

    if not time_files:
        raise FileNotFoundError(f"No time_us_output_*_*_*.csv found in {input_dir}")

    runs = defaultdict(list)

    for file_path in time_files:
        basename = os.path.basename(file_path)
        run_id, suffix = extract_run_id_and_suffix(basename)
        if run_id is None or suffix is None:
            print(f"⚠️ Skipping unrecognized file: {basename}")
            continue

        runs[run_id].append((file_path, suffix))

    print(f"Found {len(runs)} run(s): {sorted(runs.keys())}")

    for run_id, file_list in runs.items():
        print(f"\nProcessing run: {run_id}")
        # 获取列名
        first_file = file_list[0][0]
        with open(first_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            columns = [col for col in header if col != "NCU_Cycles"]

        config_data = {col: {} for col in columns}

        for file_path, suffix in file_list:
            yaml_key = FILENAME_TO_KEY[suffix]
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                times = next(reader)
            for i, col in enumerate(header):
                if col == "NCU_Cycles":
                    continue
                time_val = float(times[i])
                config_data[col][yaml_key] = time_val

        run_output_dir = os.path.join(args.output_dir, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        for config_name, data in config_data.items():
            safe_name = config_name.replace("-", "_").replace(":", "_")
            output_file = os.path.join(run_output_dir, f"{safe_name}.yaml")

            with open(output_file, 'w') as f:
                f.write("inference_term: decode\n")
                for key in ["atten_norm", "atten", "moe_norm", "moe_route", "moe_experts"]:
                    if key in data:
                        f.write(f"{key}:\n")
                        f.write(f"    time_gpu_avg: {data[key]:.3f}\n")
                    else:
                        print(f"  ⚠️ Missing {key} for config {config_name} in run {run_id}")

            print(f"  ✅ Generated: {output_file}")

if __name__ == "__main__":
    main()