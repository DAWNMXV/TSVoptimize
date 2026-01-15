import pandas as pd
import os
import glob
from config import DATA_RAW_DIR, INPUT_COLS


class DataEngine:
    @staticmethod
    def load_data_auto():
        """
        自动读取 raw 目录下所有 CSV 并合并。
        用于 Streamlit 的 cache 函数调用。
        """
        all_files = glob.glob(os.path.join(DATA_RAW_DIR, "*.csv"))

        if not all_files:
            return None, "Raw 目录下未找到 CSV 文件。"

        dfs = []
        for f in all_files:
            try:
                # 尝试多种编码读取
                try:
                    df = pd.read_csv(f, encoding='utf-8')
                except:
                    df = pd.read_csv(f, encoding='gbk')

                df.columns = df.columns.str.strip()

                # 简单校验：是否包含至少一个输入列
                if any(col in df.columns for col in INPUT_COLS):
                    # 统一数值精度
                    num_cols = df.select_dtypes(include=['float64', 'float32']).columns
                    df[num_cols] = df[num_cols].round(5)
                    dfs.append(df)
            except Exception as e:
                print(f"Skipping {f}: {e}")
                continue

        if not dfs:
            return None, "没有有效的数据文件可供合并。"

        # 拼接数据 (Union 模式，允许不同物理场文件列不同)
        df_master = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
        df_master = df_master.drop_duplicates()

        return df_master, f"成功加载 {len(all_files)} 个文件，共 {len(df_master)} 条数据。"