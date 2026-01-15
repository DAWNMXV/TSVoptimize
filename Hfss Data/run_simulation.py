# -----------------------------------------------
# 脚本: run_simulation.py (含硅基底 + 宽带扫频修复版)
# -----------------------------------------------
from pyaedt import Hfss
import traceback
import config as cfg
import os
import sys


def main():
    print(f"ℹ️ [运行中] 项目: {cfg.project_name}")

    hfss = None
    try:
        # 1. 启动 HFSS (强制显示界面以便观察)
        hfss = Hfss(project=cfg.project_name,
                    design=cfg.design_name,
                    solution_type="Modal",
                    version=cfg.hfss_version,
                    new_desktop=True,
                    non_graphical=cfg.non_graphical_post)

        print(f"✅ HFSS 已启动。")

        # ------------------------------
        # 3️⃣ 材料准备
        # ------------------------------
        print("⚙️ 正在准备材料...")
        if "silicon_dioxide" not in hfss.materials.material_keys:
            sio2 = hfss.materials.add_material("silicon_dioxide")
            sio2.permittivity = 3.9

        # ⚠️ 确保 Silicon 是电介质 (Dielectric) ⚠️
        if "silicon" in hfss.materials.material_keys:
            try:
                hfss.materials["silicon"].permittivity = 11.9
                hfss.materials["silicon"].conductivity = 0.1  # 设为0以代表高阻硅/绝缘体
                hfss.materials["silicon"].dielectric_loss_tangent = 0.01  # 加上一点损耗
            except:
                pass
        else:
            si_diel = hfss.materials.add_material("silicon")
            si_diel.permittivity = 11.9
            si_diel.conductivity = 0
            si_diel.dielectric_loss_tangent = 0.01

        print("✅ 材料准备完成")

        # ------------------------------
        # 4️⃣ 生成模型 (含硅基底逻辑)
        # ------------------------------
        hfss.modeler.model_units = "um"

        # TSV 几何参数
        R1 = cfg.r_cu
        R2 = R1 + cfg.t_sio2
        R3 = R2 + cfg.r_D
        R4 = R3 + cfg.t_sio2
        R5 = R4 + cfg.r_cu_1
        R6 = R5 + cfg.t_sio2
        radii = [R1, R2, R3, R4, R5, R6]
        # 注意: 这里的 materials 列表只用于 TSV 内部层
        materials = ["copper", "silicon_dioxide", "silicon", "silicon_dioxide", "copper", "silicon_dioxide"]
        layer_names = ["CuInner", "Oxide1", "SiCore", "Oxide2", "CuOuter", "Oxide3"]

        # 计算位置
        tsv_positions = []
        center_offset = (cfg.array_size - 1) / 2.0
        for i in range(cfg.array_size):
            for j in range(cfg.array_size):
                x = (i - center_offset) * cfg.via_pitch
                y = (j - center_offset) * cfg.via_pitch
                tsv_positions.append((x, y))

        # --- 计算边界尺寸 ---
        max_radius = R6
        pad_substrate = 40  # 硅基底保留的边距 (保持原样)
        pad_air = 100  # 🆕 新增：空气盒子比硅基底多出来的距离

        # 阵列的物理覆盖范围 (核心区域)
        min_x_core = (-center_offset) * cfg.via_pitch - max_radius
        max_x_core = (center_offset) * cfg.via_pitch + max_radius
        min_y_core = (-center_offset) * cfg.via_pitch - max_radius
        max_y_core = (center_offset) * cfg.via_pitch + max_radius

        # 2. 创建硅基底 (尺寸 = 核心 + pad_substrate)
        # 注意：Z轴还是从 0 到 via_height
        sub_min_x = min_x_core - pad_substrate
        sub_max_x = max_x_core + pad_substrate
        sub_min_y = min_y_core - pad_substrate
        sub_max_y = max_y_core + pad_substrate

        substrate = hfss.modeler.create_box(
            position=[sub_min_x, sub_min_y, 0],
            dimensions_list=[
                sub_max_x - sub_min_x,
                sub_max_y - sub_min_y,
                cfg.via_height
            ],
            name="Si_Substrate",
            material="silicon"
        )

        # 3. 创建 Airbox (尺寸 = 硅基底 + pad_air)
        # 🆕 Z轴也要加 padding，特别是上方
        # 通常下方(Z<0)如果假设理想地，可以不加；但为了辐射边界计算，上下都加比较安全。
        air_min_x = sub_min_x - pad_air
        air_max_x = sub_max_x + pad_air
        air_min_y = sub_min_y - pad_air
        air_max_y = sub_max_y + pad_air

        air_min_z = -pad_air  # 下方留空
        air_max_z = cfg.via_height + pad_air  # 上方留空

        airbox = hfss.modeler.create_box(
            position=[air_min_x, air_min_y, air_min_z],
            dimensions_list=[
                air_max_x - air_min_x,
                air_max_y - air_min_y,
                air_max_z - air_min_z
            ],
            name="Airbox",
            material="air"
        )

        print(f"✅ 尺寸更新: Substrate Pad={pad_substrate}um, Airbox Buffer={pad_air}um")

        # --- 创建 TSV 阵列 ---
        all_tsv_parts_names = []

        for idx, (x, y) in enumerate(tsv_positions):
            tsv_name_prefix = f"TSV_{idx + 1}"
            cylinders = {}

            # 从外层向内层创建圆柱
            for i in range(5, -1, -1):
                cyl_name = f"{tsv_name_prefix}_{layer_names[i]}"
                cylinders[i] = hfss.modeler.create_cylinder(
                    orientation="Z",
                    origin=[x, y, 0],
                    radius=radii[i],
                    height=cfg.via_height,
                    name=cyl_name,
                    material=materials[i]
                )

            # 消除 TSV 内部各层的重叠 (同心圆减法)
            for i in range(5, 0, -1):
                hfss.modeler.subtract(
                    blank_list=[cylinders[i].name],
                    tool_list=[cylinders[i - 1].name],
                    keep_originals=True
                )

            # 收集 TSV 最外层名称 (用于从基底中减去)
            # 注意：只需要减去最外层(Index 5)所占据的空间即可
            # 但为了保险，通常将所有金属/介质层都收集起来，或者只用最外层的 Oxide3
            # 这里我们收集所有部件，做精确减法
            for i in range(6):
                all_tsv_parts_names.append(cylinders[i].name)

        # 🆕 3. 核心布尔运算 (Boolean Subtract)
        # 逻辑A: Airbox 要挖掉 Substrate 的空间
        hfss.modeler.subtract(
            blank_list=[airbox.name],
            tool_list=[substrate.name],
            keep_originals=True
        )

        # 逻辑B: Substrate 要挖掉 TSV 的空间 (让 TSV 塞进去)
        # 这一步非常关键！否则 TSV 和 硅基底重叠，仿真会报错
        print("⚙️ 正在从硅基底中挖孔...")
        hfss.modeler.subtract(
            blank_list=[substrate.name],
            tool_list=all_tsv_parts_names,
            keep_originals=True
        )

        print("✅ 几何建模完成 (Airbox -> Substrate -> TSVs)。")

        # ------------------------------
        # 5️⃣ 设置辐射边界
        # ------------------------------
        hfss.assign_radiation_boundary_to_faces(
            airbox.faces_on_bounding_box,
            name="Rad_Boundary"
        )

        # ------------------------------
        # 6️⃣ 端口创建
        # ------------------------------
        # ... (端口创建代码保持不变，TSV位置没变) ...
        port_inner_radius = R1
        port_outer_radius = R4

        for idx, (x, y) in enumerate(tsv_positions):
            # Bottom Port
            p_b = f"P_B{idx + 1}"
            s_b = hfss.modeler.create_circle("XY", [x, y, 0], R4, name=f"S_B{idx}")
            h_b = hfss.modeler.create_circle("XY", [x, y, 0], R1, name=f"H_B{idx}")
            hfss.modeler.subtract([s_b], [h_b], keep_originals=False)
            hfss.lumped_port(s_b, integration_line=[[x + R4, y, 0], [x + R1, y, 0]], name=p_b,impedance=50,renormalize=True)   # 强制开启归一化)
            # Top Port
            p_t = f"P_T{idx + 1}"
            s_t = hfss.modeler.create_circle("XY", [x, y, cfg.via_height], R4, name=f"S_T{idx}")
            h_t = hfss.modeler.create_circle("XY", [x, y, cfg.via_height], R1, name=f"H_T{idx}")
            hfss.modeler.subtract([s_t], [h_t], keep_originals=False)
            hfss.lumped_port(s_t, integration_line=[[x + R4, y, cfg.via_height], [x + R1, y, cfg.via_height]], name=p_t,impedance=50,renormalize=True)

        # ------------------------------
        # 📌 保存模型 (防止仿真崩溃)
        # ------------------------------
        print("💾 正在保存模型...")
        hfss.save_project()

        # ------------------------------
        # 7️⃣ 仿真设置 (使用 Count Sweep 避免崩溃)
        # ------------------------------
        print("⚙️ 创建仿真设置...")
        setup = hfss.create_setup("Setup1")
        setup.props["Frequency"] = f"{cfg.sim_freq_ghz}GHz"
        setup.props["MaximumPasses"] = 6
        setup.update()

        print(f"⚙️ 创建扫频 (Points: {cfg.freq_points})...")
        hfss.create_linear_count_sweep(
            setupname="Setup1",
            unit="GHz",
            freqstart=cfg.freq_start_ghz,
            freqstop=cfg.freq_stop_ghz,
            num_of_freq_points=cfg.freq_points,
            sweepname="Sweep1",
            sweep_type="Interpolating",
            save_fields=False
        )

        # ------------------------------
        # 8️⃣ 运行仿真
        # ------------------------------
        print("🚀 开始 Analyze...")
        hfss.save_project()
        hfss.analyze_setup("Setup1")

        if not hfss.existing_analysis_sweeps:
            print("⚠️ 警告: 未检测到 Sweep 结果。")
        else:
            print("✅ 仿真成功，数据已生成。")

        hfss.save_project()
        hfss.release_desktop(close_projects=True, close_desktop=True)

    except Exception as e:
        print("\n" + "!" * 50)
        print("❌ 发生错误，HFSS 保持打开！")
        print(f"错误: {e}")
        traceback.print_exc()
        print("!" * 50 + "\n")


if __name__ == "__main__":
    main()