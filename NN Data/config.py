import os

# === 基础路径配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESS_DIR = os.path.join(BASE_DIR, 'data', 'process')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'trained')
SCALER_DIR = os.path.join(BASE_DIR, 'models', 'scaler')

for d in [DATA_RAW_DIR, DATA_PROCESS_DIR, MODEL_DIR, SCALER_DIR]:
    os.makedirs(d, exist_ok=True)

# === 核心参数定义 (新增 Frequency_GHz) ===
INPUT_COLS = [
    'Frequency_GHz',  # 新增
    'via_pitch', 'via_height', 'r_cu', 't_sio2',
    'r_D', 'r_cu_1', 'array_size'
]

# 根据新 CSV 更新输出组
OUTPUT_GROUPS = {
    'signal_integrity': ['最大插损', '所有平均插损', '最大回损', '所有平均回损'],
    # 注意：这里根据您的新 CSV 更新了列名
    'crosstalk': ['最大PS-NEXT', '最大PS-FEXT', '总的PS-NEXT', '总的PS-FEXT'],
    'multiphysics': ['T_Max_K', 'T_Avg_K', 'Stress_Mises_Glob', 'R_th_K_W']
}

# 增加频率的边界 (根据数据大致范围设定，可调整)
VARIABLE_BOUNDS = {
    'Frequency_GHz': (10.0, 80.0), # 示例范围
    'via_pitch': (20.0, 200.0),
    'via_height': (20.0, 150.0),
    'r_cu': (1.0, 15.0),
    't_sio2': (0.05, 2.0),
    'r_D': (1.0, 10.0),
    'r_cu_1': (1.0, 10.0),
    'array_size': (2.0, 10.0)
}

DEFAULT_TRAIN_CONFIG = {
    'test_size': 0.15,
    'random_state': 42,
    'epochs': 300,
    'learning_rate': 0.008,
    'batch_size': 32,
    'hidden_layers': 5,
    'neurons': 512
}