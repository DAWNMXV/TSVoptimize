import pandas as pd
import sys
import os
import shutil
import traceback
from tqdm import tqdm
import config as cfg
import run_simulation
import brun_data
import logging
from ansys.aedt.core import settings

# ---------------------------
# 日志设置 (保持原样)
settings.enable_screen_logs = False
settings.enable_file_logs = False
settings.enable_global_log_file = False
logger = logging.getLogger("ansys.aedt.core")
logger.setLevel(logging.ERROR)
# ---------------------------

# ==========================================
# 🔧 用户配置区
# ==========================================
TARGET_ROWS = list(range(1,10))


# ==========================================

def cleanup_simulation_files(run_index):
    """清理 HFSS 模型文件"""
    try:
        if os.path.exists(cfg.project_name):
            os.remove(cfg.project_name)
        results_folder = cfg.project_name + "results"
        if os.path.exists(results_folder):
            shutil.rmtree(results_folder)
        lock_file = cfg.project_name + ".lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)
    except Exception as e:
        print(f"⚠️ 清理文件错误: {e}")


def append_to_nn_database(run_index, parameter_set):
    """
    【新逻辑】
    读取 brun_data 生成的多频点汇总 CSV，
    将 LHS 参数合并到每一行，然后追加到 Master CSV。
    """
    try:
        # 1. 读取 brun_data 生成的 CSV
        if not os.path.exists(cfg.csv_run_summary):
            print(f"⚠️ Run {run_index} 未生成汇总文件，跳过入库。")
            return

        df_run = pd.read_csv(cfg.csv_run_summary)

        # 2. 将 LHS 输入参数打入每一行
        # parameter_set 是字典 { 'via_pitch': 40, ... }
        for param_key, param_val in parameter_set.items():
            df_run[param_key] = param_val

        # 添加 run_index
        df_run['run_index'] = run_index

        # 3. 追加到主训练库
        target_csv = cfg.csv_nn_training_data

        if os.path.exists(target_csv):
            try:
                # 读取旧库
                existing_df = pd.read_csv(target_csv)

                # 如果 run_index 已存在，先删除旧的 (支持重跑)
                if 'run_index' in existing_df.columns:
                    rows_before = len(existing_df)
                    existing_df = existing_df[existing_df['run_index'] != run_index]
                    if len(existing_df) < rows_before:
                        print(f"🔄 [Update] 覆盖 Run {run_index} 的旧记录...")

                # 合并
                final_df = pd.concat([existing_df, df_run], ignore_index=True)

                # 简单排序 (按 Run 再按 Freq)
                if 'Frequency_GHz' in final_df.columns:
                    final_df = final_df.sort_values(by=['run_index', 'Frequency_GHz'])

                final_df.to_csv(target_csv, index=False)

            except Exception as e:
                print(f"⚠️ 读取主库失败 ({e})，尝试直接追加模式...")
                df_run.to_csv(target_csv, mode='a', header=False, index=False)
        else:
            # 第一次创建
            df_run.to_csv(target_csv, mode='w', header=True, index=False)

        print(f"✅ Run {run_index} ({len(df_run)} 行数据) 已成功入库。")

    except Exception as e:
        print(f"❌ (Run {run_index}) 数据聚合失败: {e}")
        traceback.print_exc()


def run_single_simulation(run_index, parameter_set):
    try:
        print(f"\n{'=' * 25} 开始运行第 {run_index} 组 {'=' * 25}")
        cfg.update_params_and_paths(run_index, parameter_set)

        # 1. 仿真
        run_simulation.main()
        # 2. 后处理
        brun_data.main()
        # 3. 入库
        append_to_nn_database(run_index, parameter_set)

        return True
    except Exception as e:
        print(f"\n{'!' * 25} 第 {run_index} 组失败 {'!' * 25}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    lhs_data_file = "lhs_data.csv"

    try:
        lhs_data = pd.read_csv(lhs_data_file)
        print(f"✅ 加载 {len(lhs_data)} 组参数。")
    except Exception as e:
        print(f"❌ 无法加载 lhs_data.csv: {e}")
        sys.exit(1)

    if TARGET_ROWS:
        rows_to_process = [(r, lhs_data.iloc[r - 1]) for r in TARGET_ROWS if 1 <= r <= len(lhs_data)]
    else:
        rows_to_process = [(i + 1, row) for i, row in lhs_data.iterrows()]

    print(f"🚀 计划执行 {len(rows_to_process)} 个任务...")

    pbar = tqdm(rows_to_process, desc="LHS 进度", unit="组")

    # 记录第一个和最后一个需要处理的ID，用于保留模型
    if len(rows_to_process) > 0:
        first_run_id = rows_to_process[0][0]
        last_run_id = rows_to_process[-1][0]
    else:
        first_run_id = -1
        last_run_id = -1

    for run_index, row_series in pbar:
        parameter_set = row_series.to_dict()
        pbar.set_description(f"LHS Run {run_index}")

        if run_single_simulation(run_index, parameter_set):
            # 磁盘优化: 只保留列表中的首尾
            if run_index != first_run_id and run_index != last_run_id:
                cleanup_simulation_files(run_index)
            else:
                print(f"💾 保留模型文件 (Run {run_index})")

    print("\n🎉 任务全部完成 🎉")