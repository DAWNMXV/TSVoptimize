import numpy as np
import pandas as pd
import joblib
import os
import random
from tensorflow.keras.models import load_model
from deap import base, creator, tools
from config import INPUT_COLS, OUTPUT_GROUPS, VARIABLE_BOUNDS, SCALER_DIR
from src.model_engine import ModelEngine


class OptimizeEngine:
    def __init__(self):
        self.model_engine = ModelEngine()
        self.models = {}
        self.scalers_Y = {}
        self.scaler_X = None

        # 寻找 array_size 索引以便取整
        self.idx_array_size = -1
        if 'array_size' in INPUT_COLS:
            self.idx_array_size = INPUT_COLS.index('array_size')

        try:
            # 加载 Global Scaler
            scaler_x_path = os.path.join(SCALER_DIR, 'scaler_X.gz')
            if os.path.exists(scaler_x_path):
                self.scaler_X = joblib.load(scaler_x_path)

            # 加载模型
            for group in OUTPUT_GROUPS.keys():
                best_model_path = self.model_engine.get_best_model(group)
                scaler_y_path = os.path.join(SCALER_DIR, f'scaler_Y_{group}.gz')

                if best_model_path and os.path.exists(scaler_y_path):
                    self.models[group] = load_model(best_model_path)
                    self.scalers_Y[group] = joblib.load(scaler_y_path)
        except Exception as e:
            print(f"OptimizeEngine Init Error: {e}")

    def run_single_parameter_sweep(self, sweep_col, fixed_inputs, custom_range=None, steps=50, progress_callback=None):
        """单变量扫描"""
        if self.scaler_X is None or not self.models:
            raise ValueError("模型未完全加载，请先进行模型训练。")

        if custom_range is not None:
            low, high = custom_range
        else:
            low, high = VARIABLE_BOUNDS[sweep_col]

        x_vals = np.linspace(low, high, steps)
        if sweep_col == 'array_size':
            x_vals = np.unique(np.round(x_vals))

        results = []
        total_steps = len(x_vals)

        for idx, val in enumerate(x_vals):
            if progress_callback:
                progress_callback(idx + 1, total_steps)

            current_input = fixed_inputs.copy()
            current_input[sweep_col] = val

            try:
                X_enhanced = ModelEngine.feature_engineering_static(current_input)
                X_sc = self.scaler_X.transform(X_enhanced)

                row = current_input.copy()
                for group, model in self.models.items():
                    pred_scaled = model.predict(X_sc, verbose=0)
                    pred = self.scalers_Y[group].inverse_transform(pred_scaled)[0]
                    for i, col in enumerate(OUTPUT_GROUPS[group]):
                        v = float(pred[i])
                        if any(k in col for k in ['Stress', 'T_', 'R_th']): v = max(0.0, v)
                        row[col] = v
                results.append(row)
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def _evaluate(self, individual, targets, fixed_inputs, active_indices, output_constraints=None):
        """遗传算法评估函数 (含动态惩罚)"""
        # 1. 重构全参数
        full_params = [0.0] * len(INPUT_COLS)

        # 填入固定值
        for col, val in fixed_inputs.items():
            if col in INPUT_COLS:
                full_params[INPUT_COLS.index(col)] = val

        # 填入基因值
        for i, idx in enumerate(active_indices):
            full_params[idx] = individual[i]

        if self.idx_array_size != -1:
            full_params[self.idx_array_size] = round(full_params[self.idx_array_size])

        # 2. 预测
        input_dict = {col: val for col, val in zip(INPUT_COLS, full_params)}
        X_enhanced = ModelEngine.feature_engineering_static(input_dict)
        X_scaled = self.scaler_X.transform(X_enhanced)

        current_preds = {}
        for group, model in self.models.items():
            pred_s = model.predict(X_scaled, verbose=0)
            pred = self.scalers_Y[group].inverse_transform(pred_s)[0]
            for i, col in enumerate(OUTPUT_GROUPS[group]):
                val = float(pred[i])
                if any(k in col for k in ['Stress', 'T_', 'R_th']): val = max(0.0, val)
                current_preds[col] = val

        # 3. 计算惩罚 (Soft + Hard Penalty)
        fitness_values = []
        penalty = 0.0

        if output_constraints:
            for col, (min_v, max_v) in output_constraints.items():
                val = current_preds.get(col, 0)
                diff = 0.0
                if val < min_v:
                    diff = min_v - val
                elif val > max_v:
                    diff = val - max_v

                if diff > 0:
                    # 动态惩罚：小违规线性惩罚，大违规平方惩罚
                    if diff < 1.0:
                        penalty += diff * 100.0
                    else:
                        penalty += (diff ** 2) * 50.0
                    penalty += 10.0  # 基础违规罚分

        # 4. 计算目标
        for t_name, direction in targets.items():
            val = current_preds.get(t_name, 9999)
            if np.isnan(val): val = 9999

            # DEAP 使用 weights=(-1.0, ...) 表示最小化
            # 如果目标是 max，我们需要取负值 (-val) 喂给 DEAP
            # 如果目标是 min，我们取正值 (val)
            # Penalty 总是增加这个值 (让其变大/变差)

            if direction == 'max':
                fitness_values.append(-val + penalty)
            else:
                fitness_values.append(val + penalty)

        return tuple(fitness_values)

    def run_optimization(self, target_directions, fixed_inputs={}, output_constraints={},
                         pop_size=40, n_gen=20, progress_callback=None):
        """执行优化"""
        if self.scaler_X is None or not self.models:
            return pd.DataFrame()

        if pop_size % 4 != 0:
            pop_size = (pop_size // 4 + 1) * 4

        # 确定活跃变量
        active_vars = [c for c in INPUT_COLS if c not in fixed_inputs]
        active_indices = [INPUT_COLS.index(c) for c in active_vars]

        if not active_vars:
            # 全固定，只算一次
            temp_res = self.run_single_parameter_sweep(INPUT_COLS[0], fixed_inputs, steps=1)
            return temp_res

        # DEAP Setup
        if hasattr(creator, "FitnessMin"): del creator.FitnessMin
        if hasattr(creator, "Individual"): del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * len(target_directions))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        def create_ind():
            ind = []
            for col in active_vars:
                low, up = VARIABLE_BOUNDS[col]
                val = random.uniform(low, up)
                if col == 'array_size': val = round(val)
                ind.append(val)
            return creator.Individual(ind)

        toolbox.register("individual", create_ind)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate,
                         targets=target_directions,
                         fixed_inputs=fixed_inputs,
                         active_indices=active_indices,
                         output_constraints=output_constraints)

        # 算子
        lows = [VARIABLE_BOUNDS[c][0] for c in active_vars]
        ups = [VARIABLE_BOUNDS[c][1] for c in active_vars]
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lows, up=ups, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=lows, up=ups, eta=20.0, indpb=0.1)
        toolbox.register("select", tools.selNSGA2)

        # 运行
        pop = toolbox.population(n=pop_size)
        pop = toolbox.select(pop, len(pop))  # 初始化排序

        for gen in range(1, n_gen + 1):
            if progress_callback: progress_callback(gen, n_gen)

            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9: toolbox.mate(ind1, ind2)
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.select(pop + offspring, pop_size)

        # 提取结果
        pareto_front = tools.selNSGA2(pop, len(pop))
        results = []
        seen = set()

        for ind in pareto_front:
            gene_tuple = tuple([round(x, 4) for x in ind])
            if gene_tuple in seen: continue
            seen.add(gene_tuple)

            full_params = fixed_inputs.copy()
            for i, idx in enumerate(active_indices):
                full_params[INPUT_COLS[idx]] = ind[i]
                if INPUT_COLS[idx] == 'array_size':
                    full_params[INPUT_COLS[idx]] = round(ind[i])

            # 计算物理值
            input_dict = full_params.copy()
            X_enhanced = ModelEngine.feature_engineering_static(input_dict)
            X_sc = self.scaler_X.transform(X_enhanced)

            row = full_params.copy()
            is_valid = True

            for group, model in self.models.items():
                pred = self.scalers_Y[group].inverse_transform(model.predict(X_sc, verbose=0))[0]
                for i, col in enumerate(OUTPUT_GROUPS[group]):
                    val = pred[i]
                    if any(k in col for k in ['Stress', 'T_', 'R_th']): val = max(0.0, val)
                    row[col] = val

                    # 严格校验约束
                    if col in output_constraints:
                        min_v, max_v = output_constraints[col]
                        if not (min_v <= val <= max_v):
                            is_valid = False

            if is_valid:
                results.append(row)

        return pd.DataFrame(results)