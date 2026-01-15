import numpy as np
import pandas as pd
from ansys.aedt.core import Hfss
import traceback
import config as cfg
import os
from scipy.interpolate import interp1d  # 引入插值函数


def main():
    # ------------------------------
    # 1️⃣ 初始化
    # -----------------------------
    design_name = cfg.design_name
    project_path = cfg.project_name
    print(f"ℹ️ 后处理: {cfg.array_name_str} 阵列, 频率范围 {cfg.freq_start_ghz}-{cfg.freq_stop_ghz}GHz")

    if not os.path.exists(project_path):
        raise FileNotFoundError(f"仿真文件 {project_path} 未找到。")

    hfss = None
    processing_success = False

    try:
        print(f"⚙️ 连接 HFSS 项目: {cfg.project_name}")
        hfss = Hfss(
            project=project_path,
            design=design_name,
            solution_type="Modal",
            version=cfg.hfss_version,
            new_desktop=True,
            non_graphical=cfg.non_graphical_post
        )

        # ------------------------------
        # 2️⃣ 准备提取所有S参数
        # ------------------------------
        print("⚙️ 正在准备 S参数表达式...")

        num_ports = cfg.array_size * cfg.array_size
        bottom_ports = [f"P_B{i + 1}" for i in range(num_ports)]
        top_ports = [f"P_T{i + 1}" for i in range(num_ports)]

        all_exprs = []
        # 构建全矩阵需求
        for i in range(num_ports):  # Aggressor
            aggr_b = bottom_ports[i]
            for j in range(num_ports):  # Victim
                vic_b = bottom_ports[j]
                vic_t = top_ports[j]
                # FEXT / IL 相关 (Top vs Bottom)
                all_exprs.append(f"S({vic_t},{aggr_b})")
                # NEXT / RL 相关 (Bottom vs Bottom)
                all_exprs.append(f"S({vic_b},{aggr_b})")

        all_exprs = list(set(all_exprs))
        print(f"⚙️ 正在提取 {len(all_exprs)} 个表达式的全频段数据...")

        # ------------------------------
        # 3️⃣ 批量获取数据并插值
        # ------------------------------
        # 从 HFSS 获取原始数据 (可能是不均匀的频率点)
        data_obj = hfss.post.get_solution_data(expressions=all_exprs, setup_sweep_name="Setup1 : Sweep1")

        if not data_obj:
            raise RuntimeError("无法获取HFSS仿真结果。")

        raw_freqs = np.array(data_obj.primary_sweep_values)  # 单位通常是 GHz (根据setup) 或者 Hz
        # 确保单位统一，假设 get_solution_data 返回的是 Hz，则转为 GHz；如果是 GHz 则不变
        # PyAEDT通常返回单位一致的值。这里假设 raw_freqs 是 GHz (因为 setup 里 unit="GHz")
        # 稍微加个判断比较稳妥
        if np.mean(raw_freqs) > 1e6:
            raw_freqs = raw_freqs / 1e9  # Hz -> GHz

        # 定义目标频率 (15.0, 15.5, ..., 55.0)
        # np.arange 的 stop 是开区间，所以加 0.001 确保包含 55.0
        target_freqs = np.arange(cfg.freq_start_ghz, cfg.freq_stop_ghz + 0.001, cfg.freq_step_ghz)
        print(
            f"🎯 目标: {target_freqs[0]} - {target_freqs[-1]} GHz, 步长 {cfg.freq_step_ghz}, 共 {len(target_freqs)} 点")

        # 预计算插值数据，存入字典
        # Key: 表达式字符串, Value: 插值后的 Magnitude 数组 (长度 = len(target_freqs))
        expr_data_map = {}

        for expr in all_exprs:
            mag_data = data_obj.data_magnitude(expr)
            # 建立插值函数 (线性插值)
            f_interp = interp1d(raw_freqs, mag_data, kind='linear', fill_value="extrapolate")
            # 计算目标值
            expr_data_map[expr] = f_interp(target_freqs)

        # ------------------------------
        # 4️⃣ 内存循环计算聚合指标
        # ------------------------------
        print("📊 正在内存中计算聚合指标...")
        summary_rows = []

        # 遍历每一个目标频率点
        for idx_f, f_ghz in enumerate(target_freqs):

            # 临时列表，收集该频率下所有端口的数据
            curr_il = []
            curr_rl = []
            curr_next_totals = []
            curr_fext_totals = []

            # 遍历 Victim 端口
            for j in range(num_ports):
                vic_b = bottom_ports[j]
                vic_t = top_ports[j]

                sum_sq_next = 0.0
                sum_sq_fext = 0.0

                # 遍历 Aggressor 端口
                for i in range(num_ports):
                    aggr_b = bottom_ports[i]

                    # 从预计算字典中取值
                    mag_tb = expr_data_map[f"S({vic_t},{aggr_b})"][idx_f]
                    mag_bb = expr_data_map[f"S({vic_b},{aggr_b})"][idx_f]

                    # 避免 log(0)
                    val_tb_db = 20 * np.log10(mag_tb + 1e-15)
                    val_bb_db = 20 * np.log10(mag_bb + 1e-15)

                    if i == j:
                        # 自身: IL, RL
                        curr_il.append(val_tb_db)
                        curr_rl.append(val_bb_db)
                    else:
                        # 干扰: 累加功率 (线性值的平方)
                        sum_sq_fext += mag_tb ** 2
                        sum_sq_next += mag_bb ** 2

                # 计算该 Victim 的总串扰 (dB)
                total_fext_db = 10 * np.log10(sum_sq_fext + 1e-15)
                total_next_db = 10 * np.log10(sum_sq_next + 1e-15)

                curr_fext_totals.append(total_fext_db)
                curr_next_totals.append(total_next_db)

            # --- 汇总该频率点的统计值 ---
            row_data = {
                "Frequency_GHz": f_ghz,
                "最大插损": np.min(curr_il),
                "所有平均插损": np.mean(curr_il),
                "最大回损": np.max(curr_rl),
                "所有平均回损": np.mean(curr_rl),
                "最大PS-NEXT": np.max(curr_next_totals),  # 最差的那个端口的总串扰
                "最大PS-FEXT": np.max(curr_fext_totals),
                "总的PS-NEXT": np.mean(curr_next_totals),  # 所有端口平均的总串扰
                "总的PS-FEXT": np.mean(curr_fext_totals),

            }
            summary_rows.append(row_data)

        # ------------------------------
        # 5️⃣ 保存单一汇总 CSV
        # ------------------------------
        df_summary = pd.DataFrame(summary_rows)
        # 调整列顺序，频率在前
        cols = ['Frequency_GHz'] + [c for c in df_summary.columns if c != 'Frequency_GHz']
        df_summary = df_summary[cols]

        df_summary.to_csv(cfg.csv_run_summary, index=False)
        print(f"✅ 已保存汇总文件 (包含 {len(df_summary)} 个频点): {cfg.csv_run_summary}")

        processing_success = True

    except Exception as e:
        print(f"\n❌ 致命错误: {traceback.format_exc()}")

    finally:
        if hfss:
            hfss.close_project(save=False)
            hfss.release_desktop(close_projects=False, close_desktop=True)


if __name__ == "__main__":
    main()