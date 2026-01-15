import os
import numpy as np

# -----------------------------------------------
# --- 1. 静态参数 (仿真与频率设置) ---
# -----------------------------------------------

# 【新】求解频率 (设为最高频率以保证网格划分质量)
sim_freq_ghz = 55

# 【新】扫描范围设置
freq_start_ghz = 10.0
freq_stop_ghz = 60.0
freq_step_ghz = 0.5

# 自动计算 HFSS 扫描点数 (仅用于 HFSS 设置，实际插值在 Python 端控制)
# 我们设置足够的点数以保证 Interpolating Sweep 的精度
freq_points = int((freq_stop_ghz - freq_start_ghz) / freq_step_ghz) + 1

hfss_version = "2024.1"
non_graphical_sim = True
non_graphical_post = True

# -----------------------------------------------
# --- 2. 动态参数 (将被LHS覆盖) ---
# -----------------------------------------------
# 默认值
array_size = 5
via_pitch = 40
r_cu = 5
t_sio2 = 1
r_D = 4
r_cu_1 = 3
via_height = 110

# -----------------------------------------------
# --- 3. 动态路径 ---
# -----------------------------------------------
base_dir = os.getcwd()
sim_output_dir = ""
csv_output_dir = ""
project_name = ""
design_name = ""
array_name_str = ""

# 【新】本次运行的汇总文件 (包含该次仿真下所有频率点的数据)
csv_run_summary = ""
# 总训练数据主表
csv_nn_training_data = ""

def update_params_and_paths(run_index, params_dict):
    global array_size, via_pitch, r_cu, t_sio2, r_D, r_cu_1, via_height
    global sim_output_dir, csv_output_dir, project_name, design_name, array_name_str
    global csv_run_summary, csv_nn_training_data

    # --- A. 更新动态参数 ---
    array_size = int(params_dict.get('array_size', array_size))
    via_pitch = params_dict.get('via_pitch', via_pitch)
    r_cu = params_dict.get('r_cu', r_cu)
    t_sio2 = params_dict.get('t_sio2', t_sio2)
    r_D = params_dict.get('r_D', r_D)
    r_cu_1 = params_dict.get('r_cu_1', r_cu_1)
    via_height = params_dict.get('via_height', via_height)

    # --- B. 定义父文件夹 ---
    sweep_parent_folder = os.path.join(base_dir, "LHS_Sweep_Results")
    os.makedirs(sweep_parent_folder, exist_ok=True)
    csv_nn_training_data = os.path.join(sweep_parent_folder, "NN_Training_Data_Master.csv")

    # --- C. 定义本次运行的特定文件夹 ---
    run_folder_name = f"Run_{run_index:04d}"
    sim_output_dir = os.path.join(sweep_parent_folder, run_folder_name)
    os.makedirs(sim_output_dir, exist_ok=True)

    csv_folder_name = "计算结果"
    csv_output_dir = os.path.join(sim_output_dir, csv_folder_name)
    os.makedirs(csv_output_dir, exist_ok=True)

    # --- D. 重新生成路径 ---
    array_name_str = f"{array_size}x{array_size}"
    project_name = os.path.join(sim_output_dir, f"TSV_Array_{array_name_str}_Modal.aedt")
    design_name = f"TSV_6Layer_Array_{array_name_str}"

    # 【新】只定义一个汇总文件路径 (取代了之前分散的6个CSV)
    csv_run_summary = os.path.join(csv_output_dir, f"Summary_Run_{run_index}_{array_name_str}.csv")

# 初始化 (Run 0 仅用于测试路径生成)
default_params = {
    'array_size': array_size, 'via_pitch': via_pitch, 'r_cu': r_cu,
    't_sio2': t_sio2, 'r_D': r_D, 'r_cu_1': r_cu_1, 'via_height': via_height
}
update_params_and_paths(0, default_params)