import os
import joblib
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import MODEL_DIR, SCALER_DIR, INPUT_COLS, OUTPUT_GROUPS, DATA_PROCESS_DIR


class ModelEngine:
    def __init__(self):
        # 确保每个类别的模型文件夹存在
        for group in OUTPUT_GROUPS.keys():
            os.makedirs(os.path.join(MODEL_DIR, group), exist_ok=True)
        os.makedirs(SCALER_DIR, exist_ok=True)

    @staticmethod
    def feature_engineering_static(df_or_dict):
        """
        静态特征工程，增加物理衍生变量 (Physics-Informed)
        """
        if isinstance(df_or_dict, dict):
            df = pd.DataFrame([df_or_dict])
        elif isinstance(df_or_dict, (list, np.ndarray)):
            df = pd.DataFrame(np.array(df_or_dict).reshape(1, -1), columns=INPUT_COLS)
        else:
            df = df_or_dict.copy()

        # 补全缺失列以防报错
        for col in INPUT_COLS:
            if col not in df.columns:
                df[col] = 0.0

        # === 提取基础物理量 ===
        # 确保数据类型为 float
        for col in INPUT_COLS:
            df[col] = df[col].astype(float)

        r_cu = df['r_cu'].values
        t_sio2 = df['t_sio2'].values
        r_D = df['r_D'].values
        via_height = df['via_height'].values
        via_pitch = df['via_pitch'].values
        freq = df['Frequency_GHz'].values  # 新增：频率

        epsilon = 1e-6

        # === 构造几何物理特征 ===
        feat_area_cu = np.pi * (r_cu ** 2)
        feat_inv_r_cu = 1.0 / (r_cu + epsilon)
        feat_inv_dist = 1.0 / (r_D + epsilon)
        feat_aspect_ratio = via_height / (r_cu + epsilon)
        feat_insulator_ratio = t_sio2 / (r_cu + epsilon)
        diameter = 2 * (r_cu + t_sio2)
        feat_pitch_ratio = via_pitch / (diameter + epsilon)

        # === 新增：电磁物理特征 ===
        # 集肤效应项 ~ sqrt(f)
        feat_skin_effect = np.sqrt(freq + epsilon)
        # 简单的介质损耗项 ~ f * t_sio2
        feat_dielectric_loss = freq * t_sio2

        X_raw = df[INPUT_COLS].values

        # 拼接增强特征 (原始输入 + 物理特征)
        X_enhanced = np.column_stack((
            X_raw,
            feat_area_cu, feat_inv_r_cu, feat_inv_dist,
            feat_aspect_ratio, feat_insulator_ratio, feat_pitch_ratio,
            feat_skin_effect, feat_dielectric_loss
        ))

        return X_enhanced

    def fit_and_save_global_scaler(self, df_master):
        """
        【关键修改】构建并锁定全局 Scaler_X
        确保所有物理场模型使用完全相同的输入分布。
        """
        print("正在构建全局输入特征 Scaler...")
        # 过滤掉完全缺失输入的行
        df_clean = df_master.dropna(subset=INPUT_COLS)
        X_enhanced = self.feature_engineering_static(df_clean)

        scaler_X = StandardScaler().fit(X_enhanced)
        joblib.dump(scaler_X, os.path.join(SCALER_DIR, 'scaler_X.gz'))
        print("全局 Scaler 已保存。")
        return scaler_X

    def _build_dynamic_model(self, input_dim, output_dim, config):
        """动态构建模型"""
        model = Sequential()
        model.add(Dense(config['neurons'], activation='relu', input_shape=(input_dim,)))
        model.add(BatchNormalization())

        for _ in range(config['hidden_layers'] - 1):
            model.add(Dense(config['neurons'], activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))  # 防止过拟合

        model.add(Dense(output_dim))

        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        return model

    def _manage_top_models(self, group_name, new_model, score):
        """管理 Top 10 模型"""
        group_dir = os.path.join(MODEL_DIR, group_name)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{score:.5f}_{timestamp}.keras"
        save_path = os.path.join(group_dir, filename)

        new_model.save(save_path)

        # 扫描并清理旧模型
        files = glob.glob(os.path.join(group_dir, "*.keras"))
        model_list = []
        for f in files:
            try:
                basename = os.path.basename(f)
                s = float(basename.split('_')[0])
                model_list.append((s, f))
            except:
                continue

        model_list.sort(key=lambda x: x[0], reverse=True)

        if len(model_list) > 10:
            for _, f_path in model_list[10:]:
                os.remove(f_path)

        return len(model_list)

    def get_best_model(self, group_name):
        """获取分数最高的模型"""
        group_dir = os.path.join(MODEL_DIR, group_name)
        files = glob.glob(os.path.join(group_dir, "*.keras"))
        if not files: return None

        best_file = None
        best_score = -9999.0

        for f in files:
            try:
                basename = os.path.basename(f)
                s = float(basename.split('_')[0])
                if s > best_score:
                    best_score = s
                    best_file = f
            except:
                continue
        return best_file

    def _calculate_raw_feature_importance(self, model, scaler_X, df_sample, output_cols):
        """
        【新增】通过置换(Permutation)计算原始输入的重要性百分比
        """
        # 1. 计算基准预测
        baseline_X_enhanced = self.feature_engineering_static(df_sample)
        baseline_pred = model.predict(scaler_X.transform(baseline_X_enhanced), verbose=0)

        # 计算基准的总变异 (Variance)，用于衡量模型波动幅度
        # 加上微小值防止除零
        total_variance = np.var(baseline_pred, axis=0).sum() + 1e-9

        importances = {}

        # 2. 逐个打乱原始输入列
        for col in INPUT_COLS:
            df_shuffled = df_sample.copy()
            # 随机打乱当前列，破坏该特征的信息
            df_shuffled[col] = np.random.permutation(df_shuffled[col].values)

            # 重新特征工程 + 预测
            X_shuffled = self.feature_engineering_static(df_shuffled)
            pred_shuffled = model.predict(scaler_X.transform(X_shuffled), verbose=0)

            # 计算 MSE 差异 (Mean Squared Error Diff)
            # 差异越大，说明该特征越重要
            mse_diff = np.mean((baseline_pred - pred_shuffled) ** 2)
            importances[col] = mse_diff

        # 3. 归一化为百分比
        total_imp = sum(importances.values()) + 1e-9
        importance_percent = {k: (v / total_imp) * 100 for k, v in importances.items()}

        return importance_percent

    def train_custom_model(self, df_master, group_name, config):
        """训练入口"""
        output_cols = OUTPUT_GROUPS[group_name]
        valid_cols = [c for c in output_cols if c in df_master.columns]

        if not valid_cols:
            return 0.0, 0.0, "该组数据列不存在", {}

        # 准备数据
        df_subset = df_master.dropna(subset=INPUT_COLS + valid_cols)
        if len(df_subset) < 10:
            return 0.0, 0.0, "样本过少", {}

        # === 1. 加载全局 Scaler (不重新训练) ===
        scaler_path = os.path.join(SCALER_DIR, 'scaler_X.gz')
        if not os.path.exists(scaler_path):
            # 兜底逻辑：如果全局Scaler不存在，强制生成一次 (针对初次运行)
            print("Warning: 全局 Scaler 未找到，正在使用当前全集生成...")
            self.fit_and_save_global_scaler(df_master)

        scaler_X = joblib.load(scaler_path)
        X_enhanced = self.feature_engineering_static(df_subset)
        X_scaled = scaler_X.transform(X_enhanced)

        # 2. Y Scaler (每个组独立)
        Y = df_subset[valid_cols].values
        scaler_Y = StandardScaler().fit(Y)
        joblib.dump(scaler_Y, os.path.join(SCALER_DIR, f'scaler_Y_{group_name}.gz'))
        Y_scaled = scaler_Y.transform(Y)

        # 3. 划分
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y_scaled, test_size=config['test_size'], random_state=42
        )

        # 4. 构建与训练
        model = self._build_dynamic_model(X_train.shape[1], Y_train.shape[1], config)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        ]

        model.fit(
            X_train, Y_train,
            validation_split=0.1,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            verbose=0,
            callbacks=callbacks
        )

        # 5. 评估
        score = 0.0
        mae = 0.0
        importance_dict = {}

        if len(X_test) > 0:
            Y_pred = scaler_Y.inverse_transform(model.predict(X_test, verbose=0))
            Y_true = scaler_Y.inverse_transform(Y_test)
            score = r2_score(Y_true, Y_pred) * 100
            mae = mean_absolute_error(Y_true, Y_pred)
        if config.get('enable_xai', False):
            print("正在计算特征重要性 (可能会慢)...")
            # 计算重要性 (使用部分测试集加速)
            sample_size = min(200, len(X_test))
            sample_df = df_subset.iloc[:sample_size].copy()
            importance_dict = self._calculate_raw_feature_importance(model, scaler_X, sample_df, valid_cols)
        else:
            importance_dict = {}  # 不计算，直接返回空
        # 保存模型
        self._manage_top_models(group_name, model, score)

        return score, mae, "Success", importance_dict

    def batch_predict(self, df_input):
        """
        批量预测接口：接收一个 DataFrame，返回带预测结果的 DataFrame
        """
        # 1. 确保包含所有必要的输入列 (缺失填0)
        df_process = df_input.copy()
        for col in INPUT_COLS:
            if col not in df_process.columns:
                df_process[col] = 0.0

        try:
            # 2. 加载 X Scaler 并进行特征工程
            scaler_x_path = os.path.join(SCALER_DIR, 'scaler_X.gz')
            if not os.path.exists(scaler_x_path):
                return None, "Scaler 未找到，请先训练模型"

            scaler_X = joblib.load(scaler_x_path)

            # 批量特征工程
            X_enhanced = self.feature_engineering_static(df_process)
            X_scaled = scaler_X.transform(X_enhanced)

            # 3. 遍历所有物理场模型进行预测
            results_df = df_input.copy()  # 结果表保留原始输入

            for group_name, cols in OUTPUT_GROUPS.items():
                best_model_path = self.get_best_model(group_name)
                scaler_y_path = os.path.join(SCALER_DIR, f'scaler_Y_{group_name}.gz')

                if best_model_path and os.path.exists(scaler_y_path):
                    # 加载模型和 Y Scaler
                    model = load_model(best_model_path)
                    scaler_Y = joblib.load(scaler_y_path)

                    # 批量预测 (速度很快)
                    pred_scaled = model.predict(X_scaled, verbose=0)
                    pred_original = scaler_Y.inverse_transform(pred_scaled)

                    # 将结果填回 DataFrame
                    for i, col in enumerate(cols):
                        # 物理约束修正 (非负)
                        if any(k in col for k in ['Stress', 'T_', 'R_th']):
                            pred_original[:, i] = np.maximum(0.0, pred_original[:, i])

                        results_df[col] = pred_original[:, i]
                else:
                    # 如果该物理场没有模型，填 NaN
                    for col in cols:
                        results_df[col] = None

            return results_df, "批量预测成功"

        except Exception as e:
            return None, str(e)

    def predict_and_save(self, input_dict):
        """预测并保存到历史记录"""
        try:
            X_enhanced = self.feature_engineering_static(input_dict)
            scaler_x_path = os.path.join(SCALER_DIR, 'scaler_X.gz')

            if not os.path.exists(scaler_x_path):
                return None, "Scaler未找到，请先训练"

            scaler_X = joblib.load(scaler_x_path)
            X_scaled = scaler_X.transform(X_enhanced)

            results = input_dict.copy()

            for group_name, cols in OUTPUT_GROUPS.items():
                best_model_path = self.get_best_model(group_name)
                scaler_y_path = os.path.join(SCALER_DIR, f'scaler_Y_{group_name}.gz')

                if best_model_path and os.path.exists(scaler_y_path):
                    model = load_model(best_model_path)
                    scaler_Y = joblib.load(scaler_y_path)

                    pred_s = model.predict(X_scaled, verbose=0)
                    pred = scaler_Y.inverse_transform(pred_s)[0]

                    for i, col in enumerate(cols):
                        if i < len(pred):
                            val = float(pred[i])
                            if any(k in col for k in ['Stress', 'T_', 'R_th']):
                                val = max(0.0, val)
                            results[col] = val
                else:
                    for col in cols:
                        results[col] = None

            # 保存
            save_file = os.path.join(DATA_PROCESS_DIR, 'prediction_history.csv')
            df_res = pd.DataFrame([results])

            if not os.path.exists(save_file):
                df_res.to_csv(save_file, index=False, encoding='utf-8-sig')
            else:
                df_res.to_csv(save_file, mode='a', header=False, index=False, encoding='utf-8-sig')

            return results, "预测成功并已保存"
        except Exception as e:
            return None, str(e)
