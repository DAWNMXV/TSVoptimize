import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from config import INPUT_COLS, OUTPUT_GROUPS, DEFAULT_TRAIN_CONFIG, VARIABLE_BOUNDS, DATA_PROCESS_DIR, SCALER_DIR
from src.data_engine import DataEngine
from src.model_engine import ModelEngine
from src.optimize_engine import OptimizeEngine

st.set_page_config(page_title="CTSV 智能优化平台 Pro", layout="wide", page_icon="🧬")

# === 全局 State 初始化 ===
if 'train_config' not in st.session_state:
    st.session_state['train_config'] = DEFAULT_TRAIN_CONFIG.copy()
if 'sweep_result' not in st.session_state:
    st.session_state['sweep_result'] = None
if 'last_importance' not in st.session_state:
    st.session_state['last_importance'] = None


# === 核心资源加载 (Cached) ===
@st.cache_data
def load_data_cached():
    return DataEngine.load_data_auto()


# 缓存优化引擎 (避免每次交互重载模型)
@st.cache_resource
def get_optimize_engine():
    return OptimizeEngine()


# 侧边栏
st.sidebar.title("🧬 CTSV Pro")
page = st.sidebar.radio("功能模块",
                        ["1. 数据概览", "2. 模型训练 (调优)", "3. 性能预测", "4. 单参数敏感度分析",
                         "5. 多目标协同优化"])

# 加载数据
df_master, msg = load_data_cached()

# === 系统初始化检查：构建全局 Scaler ===
if df_master is not None:
    scaler_check_path = os.path.join(SCALER_DIR, 'scaler_X.gz')
    if not os.path.exists(scaler_check_path):
        st.toast("检测到系统初次运行，正在构建全局特征归一化器 (Global Scaler)...")
        # 临时实例化一个 ModelEngine 来构建 Scaler
        tmp_engine = ModelEngine()
        tmp_engine.fit_and_save_global_scaler(df_master)
        st.toast("全局 Scaler 构建完成！")

# ==========================================
# 1. 数据概览
# ==========================================
if page == "1. 数据概览":
    st.title("📂 数据资产管理")
    if df_master is not None:
        st.success(msg)
        c1, c2, c3 = st.columns(3)
        c1.metric("样本总数", len(df_master))
        c2.metric("输入特征数", len(INPUT_COLS))
        c3.metric("物理场组数", len(OUTPUT_GROUPS))

        with st.expander("查看原始数据前50行"):
            st.dataframe(df_master.head(50))

        with st.expander("数据分布可视化"):
            feat = st.selectbox("选择特征查看分布", INPUT_COLS + list(df_master.columns))
            fig = px.histogram(df_master, x=feat, nbins=30, title=f"{feat} 分布")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"数据加载失败: {msg}")

# ==========================================
# 2. 模型训练 (调优)
# ==========================================
elif page == "2. 模型训练 (调优)":
    st.title("🧠 神经网络训练工厂 (Physics-Informed)")

    if df_master is None:
        st.warning("无数据")
        st.stop()

    with st.expander("⚙️ 超参数配置", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        epochs = c1.number_input("Epochs", 50, 2000, 200, step=50)
        lr = c2.number_input("Learning Rate", 1e-5, 0.1, 0.005, format="%.5f")
        layers = c3.slider("Hidden Layers", 1, 10, 3)
        neurons = c4.slider("Neurons per Layer", 16, 512, 128, step=16)
        enable_xai = st.checkbox("计算特征重要性 (XAI) - ⚠️ 耗时操作，建议调试时关闭", value=False)

        config = {
            'epochs': epochs, 'learning_rate': lr,
            'hidden_layers': layers, 'neurons': neurons,
            'batch_size': 32, 'test_size': 0.15,
            'enable_xai': enable_xai  # 将开关状态传入配置
        }



    st.divider()
    col_main, col_info = st.columns([2, 1])

    with col_main:
        st.subheader("选择训练任务")
        selected_groups = st.multiselect("选择物理场模型", list(OUTPUT_GROUPS.keys()),
                                         default=list(OUTPUT_GROUPS.keys()))

        if st.button("🚀 开始训练与优选", type="primary"):
            engine = ModelEngine()
            progress_bar = st.progress(0)
            status_text = st.empty()
            total = len(selected_groups)

            # 清除之前的优化引擎缓存，因为模型变了
            st.cache_resource.clear()

            for i, group in enumerate(selected_groups):
                status_text.text(f"正在训练: {group} ...")
                # 接收4个返回值，包含 importance
                score, mae, msg, importance = engine.train_custom_model(df_master, group, config)

                st.session_state[f'res_{group}'] = (score, mae, msg)
                st.session_state['last_importance'] = importance  # 保存最后一个任务的重要性供展示

                progress_bar.progress((i + 1) / total)

            st.success("所有任务完成！")

            # 展示结果
            for group in selected_groups:
                if f'res_{group}' in st.session_state:
                    s, m, _ = st.session_state[f'res_{group}']
                    st.metric(f"{group} R²", f"{s:.2f}%", delta=f"MAE: {m:.4f}")

    # === 展示可解释性 (XAI) ===
if st.session_state['last_importance']:
        st.divider()
        st.subheader("📊 模型可解释性分析 (XAI)")
        st.caption("基于置换重要性 (Permutation Importance) 计算的输入参数贡献度百分比")

        imp_data = st.session_state['last_importance']
        df_imp = pd.DataFrame(list(imp_data.items()), columns=['Feature', 'Contribution (%)'])
        df_imp = df_imp.sort_values(by='Contribution (%)', ascending=True)

        fig_imp = px.bar(df_imp, x='Contribution (%)', y='Feature', orientation='h',
                         title="Parameter Contribution Analysis",
                         text_auto='.1f',
                         color='Contribution (%)', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)

# ==========================================
# 3. 性能预测
# ==========================================
elif page == "3. 性能预测":
    st.title("🔮 快速性能预测")

    tab1, tab2 = st.tabs(["⚡ 单点快速预测", "📂 批量文件预测"])

    # === Tab 1: 原有的手动输入功能 ===
    with tab1:
        with st.form("predict_form"):
            st.subheader("输入设计参数")
            inputs = {}
            cols = st.columns(4)
            for i, col_name in enumerate(INPUT_COLS):
                with cols[i % 4]:
                    default_v = float(VARIABLE_BOUNDS.get(col_name, (0.0, 0.0))[0])
                    inputs[col_name] = st.number_input(f"{col_name}", value=default_v, format="%.2f")
            submit = st.form_submit_button("计算")

        if submit:
            engine = ModelEngine()
            res, msg = engine.predict_and_save(inputs)
            if res:
                st.success(msg)
                st.json(res)  # 简单展示
            else:
                st.error(f"预测失败: {msg}")

    # === Tab 2: 新增的 CSV 批量预测功能 ===
    with tab2:
        st.subheader("上传参数 CSV 文件")
        st.info(f"请上传包含以下列的 CSV: {', '.join(INPUT_COLS)}")

        uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])

        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            st.write("预览上传数据:", df_input.head())

            if st.button("开始批量预测", type="primary"):
                engine = ModelEngine()
                with st.spinner("正在调用神经网络进行批量推理..."):
                    df_result, msg = engine.batch_predict(df_input)

                if df_result is not None:
                    st.success(msg)
                    st.dataframe(df_result)

                    # 提供下载按钮
                    csv = df_result.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下载预测结果 CSV",
                        data=csv,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"预测出错: {msg}")

# ==========================================
# 4. 单参数敏感度分析
# ==========================================
elif page == "4. 单参数敏感度分析":
    st.title("📈 敏感度扫描")

    c1, c2 = st.columns([1, 3])
    with c1:
        target_var = st.selectbox("扫描变量 (X轴)", INPUT_COLS)
        default_min, default_max = VARIABLE_BOUNDS[target_var]

        c_min, c_max = st.columns(2)
        scan_min = c_min.number_input("Min", value=float(default_min), format="%.2f")
        scan_max = c_max.number_input("Max", value=float(default_max), format="%.2f")
        steps = st.slider("Points", 10, 100, 30)

        st.subheader("固定其他参数")
        fixed_inputs_sweep = {}
        for col in INPUT_COLS:
            if col != target_var:
                val = st.number_input(f"{col}", value=float(VARIABLE_BOUNDS[col][0]), key=f"sw_fix_{col}")
                fixed_inputs_sweep[col] = val

        if st.button("开始分析", type="primary"):
            opt_engine = get_optimize_engine()
            df_res = opt_engine.run_single_parameter_sweep(
                target_var, fixed_inputs_sweep, custom_range=(scan_min, scan_max), steps=steps
            )
            st.session_state['sweep_result'] = df_res
            st.session_state['sweep_target'] = target_var

    with c2:
        if st.session_state['sweep_result'] is not None:
            df_res = st.session_state['sweep_result']
            x_col = st.session_state['sweep_target']

            all_outputs = []
            for v in OUTPUT_GROUPS.values(): all_outputs.extend(v)
            y_targets = st.multiselect("观察指标 (Y轴)", all_outputs, default=[all_outputs[0]])

            if y_targets:
                fig = px.line(df_res, x=x_col, y=y_targets, markers=True, title=f"Sensitivity: {x_col}")
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 多目标协同优化
# ==========================================
elif page == "5. 多目标协同优化":
    st.title("🎯 电-热-力协同优化 (Pareto)")

    all_outputs = []
    for p in OUTPUT_GROUPS.values(): all_outputs.extend(p)

    with st.expander("⚙️ 优化问题定义", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("1. 固定输入")
            fixed_inputs = {}
            for col_name in INPUT_COLS:
                # 使用 checkbox 决定是否固定
                is_fixed = st.checkbox(f"Lock {col_name}", key=f"opt_chk_{col_name}")
                if is_fixed:
                    val = st.number_input(f"Value", value=float(VARIABLE_BOUNDS[col_name][0]),
                                          key=f"opt_val_{col_name}")
                    fixed_inputs[col_name] = val

        with col2:
            st.subheader("2. 输出约束")
            constraints = {}
            sel_cons = st.multiselect("添加约束", all_outputs)
            for t in sel_cons:
                c_a, c_b = st.columns(2)
                min_v = c_a.number_input(f"{t} Min", -9999.0, key=f"min_{t}")
                max_v = c_b.number_input(f"{t} Max", 9999.0, key=f"max_{t}")
                constraints[t] = (min_v, max_v)

        with col3:
            st.subheader("3. 优化目标")
            sel_objs = st.multiselect("选择目标 (Pareto Axis)", all_outputs, default=[all_outputs[0]])
            target_directions = {}
            for t in sel_objs:
                mode = st.radio(f"{t} 方向", ["Min", "Max"], key=f"dir_{t}")
                target_directions[t] = 'min' if mode == "Min" else 'max'

    if st.button("🧬 启动进化算法 (NSGA-II)", type="primary"):
        if len(target_directions) < 1:
            st.error("至少选择一个优化目标")
        else:
            opt_engine = get_optimize_engine()

            progress_bar = st.progress(0)
            status_text = st.empty()


            def update_prog(gen, total):
                progress_bar.progress(gen / total)
                status_text.text(f"Evolution Generation: {gen}/{total}")


            df_res = opt_engine.run_optimization(
                target_directions, fixed_inputs, constraints,
                pop_size=40, n_gen=30, progress_callback=update_prog
            )

            if not df_res.empty:
                st.success(f"优化完成，发现 {len(df_res)} 个非支配解。")

                # === 高级 Pareto 可视化 ===
                if len(sel_objs) >= 2:
                    st.subheader("🌐 高维 Pareto 前沿可视化")

                    x_axis = sel_objs[0]
                    y_axis = sel_objs[1]
                    # 尝试自动分配颜色和大小维度
                    c_axis = sel_objs[2] if len(sel_objs) > 2 else INPUT_COLS[0]
                    s_axis = sel_objs[3] if len(sel_objs) > 3 else None

                    fig = px.scatter(
                        df_res, x=x_axis, y=y_axis,
                        color=c_axis, size=s_axis,
                        hover_data=INPUT_COLS,
                        title=f"Pareto Front: {x_axis} vs {y_axis}",
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df_res)
            else:
                st.warning("未找到满足约束的解，请放宽条件。")