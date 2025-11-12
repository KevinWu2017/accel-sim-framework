#!/usr/bin/env python3
import os
import re
import csv
import argparse
from collections import defaultdict, OrderedDict

def parse_sim_file(filepath):
    """
    解析单个 A100_*.csv 文件，提取 gpu_tot_sim_cycle 段下的 (benchmark,args)->value 映射。
    该文件是按照段落 (separator line of dashes), 标题行 (regex string), APPS,..., 然后多行 key,value 的格式组织的。
    我们要定位到标题行 "gpu_tot_sim_cycle\s*=\s*(.*)," 然后读取后面的 key,value 行直到下一个分隔段。
    返回 dict: { "Bench,Args": float_value, ... }
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()

    results = {}
    # 标题匹配：确切匹配包含 gpu_tot_sim_cycle 的标题行（可能带其它字符）
    title_re = re.compile(r"^gpu_tot_sim_cycle\\s*\=\s*\(?.*?\)\,?$|^gpu_tot_sim_cycle\s*\\s*\=\s*\(?.*?\)\,?$", re.IGNORECASE)
    # But based on your files the title line often literally is: gpu_tot_sim_cycle\s*=\s*(.*),
    # so allow a fuzzy match by checking substring "gpu_tot_sim_cycle"
    for idx, line in enumerate(lines):
        if "gpu_tot_sim_cycle" in line:
            # Found the marker line — now read subsequent lines until next separator (line starting with ----) or empty
            j = idx + 1
            # skip optional APPS line if present
            # advance until we hit a non-empty, non-sep line that looks like "APPS,..."
            # then next lines are the key,value pairs
            # Continue until next separator line that consists mostly of '-' or a blank line followed by a '----' block
            while j < len(lines):
                cur = lines[j].strip()
                # stop if we hit a separator line (many '-' characters) or an empty "----," style line
                if cur.startswith("----") or (cur == "" and (j+1 < len(lines) and lines[j+1].strip().startswith("----"))):
                    break
                # skip APPS header lines
                if cur.upper().startswith("APPS") or cur == "":
                    j += 1
                    continue
                # Expect lines of the form: Key,Value
                if ',' in cur:
                    # Some lines include trailing commas or spaces; split only on first comma
                    key, val = cur.split(',', 1)
                    key = key.strip()
                    val = val.strip().rstrip(',')
                    # Only accept keys that look like "Bench/args--..."
                    if '--' in key or '/' in key:
                        # Extract bench and args part
                        try:
                            bench, args = key.split('/', 1)
                            args = args.split('--')[0]
                            # convert numeric
                            try:
                                num = float(val)
                            except:
                                # if value is non-numeric, skip
                                j += 1
                                continue
                            results[f"{bench},{args}"] = num
                        except Exception:
                            # malformed key, skip
                            pass
                j += 1
            # If file may contain multiple gpu_tot_sim_cycle blocks, continue searching further blocks
    return results

def parse_ncu_file(ncu_csv):
    """解析 Nsight Compute 结果文件，提取 Elapsed Cycles"""
    if not ncu_csv or not os.path.exists(ncu_csv):
        print("⚠️ 未提供 --ncu_results 或文件不存在，跳过 real_gpu_ncu 数据。")
        return {}

    ncu_data = {}
    with open(ncu_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench = row.get('benchmark', '').strip()
            args = row.get('args', '').strip().replace(' ', '_')  # <--- 关键修改
            metric = row.get('metric_name', '').strip()
            val = row.get('metric_value', '').strip()
            if bench and args and metric == 'Elapsed Cycles' and val:
                try:
                    ncu_data[f"{bench},{args}"] = float(val)
                except ValueError:
                    continue
    print(f"✅ 已加载 {len(ncu_data)} 条 real_gpu_ncu 数据。")
    return ncu_data


def natural_sort_sim_names(sim_names):
    """将列名按 baseline + 延迟数值顺序排序"""
    def sort_key(name):
        # baseline
        if name == "A100-SASS":
            return (0, 0)
        # 匹配 latency 数值
        m = re.search(r"Latency\d+_(\d+)", name)
        if m:
            return (1, int(m.group(1)))  # 数值越小优先
        else:
            return (2, name)  # 无匹配的放最后按字母
    return sorted(sim_names, key=sort_key)


def collect_results(sim_dir, ncu_path=None):
    sim_dir = os.path.expanduser(sim_dir)
    files = sorted([f for f in os.listdir(sim_dir) if f.endswith('.csv') and f.startswith('A100')])
    if not files:
        print(f"❌ 未在目录 {sim_dir} 中找到任何 A100_*.csv 文件。")
        return

    all_data = defaultdict(lambda: OrderedDict())
    processed_files = 0

    for f in files:
        sim_name = os.path.splitext(f)[0]
        filepath = os.path.join(sim_dir, f)
        sim_data = parse_sim_file(filepath)
        processed_files += 1
        if not sim_data:
            print(f"⚠️ 文件 {f} 中未找到 gpu_tot_sim_cycle 数据，跳过。")
            continue
        for key, val in sim_data.items():
            all_data[key][sim_name] = val

    # 加载 NCU 数据
    ncu_data = parse_ncu_file(ncu_path) if ncu_path else {}

    # 分组
    grouped = defaultdict(list)
    for key, val_dict in all_data.items():
        try:
            bench, args = key.split(',', 1)
        except ValueError:
            continue
        grouped[bench].append((args, val_dict))

    # 写出最终表格
    output_path = os.path.join(sim_dir, "final_results.csv")
    with open(output_path, 'w', newline='') as out:
        writer = csv.writer(out)

        for i, (bench, records) in enumerate(sorted(grouped.items())):
            writer.writerow([f"### {bench}"])
            all_sims = natural_sort_sim_names({sim for _, vals in records for sim in vals.keys()})
            header = ['benchmark', 'args', 'real_gpu_ncu'] + all_sims
            writer.writerow(header)

            for args, vals in sorted(records, key=lambda x: x[0]):
                key = f"{bench},{args}"
                real_val = ncu_data.get(key, '')
                row = [bench, args, real_val]
                for sim in all_sims:
                    row.append(vals.get(sim, ''))
                writer.writerow(row)

            writer.writerow([])
            writer.writerow(['---'])
            writer.writerow([])

    print(f"✅ 汇总完成：{output_path}")
    print(f"📊 共处理 {processed_files} 个结果文件，生成 {len(grouped)} 个程序对比表。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Accel-Sim results and real GPU NCU results into one CSV.")
    parser.add_argument("--sim_results", required=True, help="Path to the simulation results directory.")
    parser.add_argument("--ncu_results", help="Path to the NCU Elapsed Cycles CSV file.")
    args = parser.parse_args()

    collect_results(args.sim_results, args.ncu_results)