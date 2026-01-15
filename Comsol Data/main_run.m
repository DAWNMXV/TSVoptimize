function main_run()
    % ================= CONFIGURATION =================
    target_rows = [1];            % 目标行
    modelDir    = 'saved_models_sweep_auto';
    csv_input   = 'lhs_data.csv';
    csv_output  = 'dataset_output_sweep_auto.csv';
    
    global_Q0   = 3e7;            
    array_N     = 5;              
    
    % --- 网格配置 (修改后) ---
    % XY平面网格密度：1=极细, 5=正常, 6=较粗, 7=粗化
    % 由于保留了 Sweep，Z轴精度由 layers_tsv 保证，XY平面可以用较粗网格来节省计算量
    mesh_cfg.h_auto_level = 4;    
    
    % Z轴层数控制 (Sweep)
    mesh_cfg.layers_buffer = 2;   % 缓冲层层数
    mesh_cfg.layers_tsv    = 5;   % TSV中间段层数
    % =================================================

    if ~exist(modelDir, 'dir'), mkdir(modelDir); end
    if ~isfile(csv_input), error('未找到输入文件: %s', csv_input); end
    
    data = load_data_robust(csv_input, csv_output);
    
    % 材料库
    db.Si   = [130,  2330, 700, 170e9, 0.28, 2.6e-6]; 
    db.SiO2 = [1.4,  2200, 730, 70e9,  0.17, 0.5e-6];
    db.Cu   = [400,  8960, 385, 110e9, 0.34, 17e-6];  
    
    fprintf('>> 任务开始: 自动扫掠网格版 (Auto-Sweep Mesh)\n');

    for k = 1:length(target_rows)
        idx = target_rows(k);
        if idx > height(data), continue; end
        
        t_start = tic;
        
        % 组装参数
        p = struct();
        p.vals.r_cu   = data.r_cu(idx);   
        p.vals.t_sio2 = data.t_sio2(idx);
        p.vals.r_D    = data.r_D(idx);    
        p.vals.r_cu_1 = data.r_cu_1(idx);
        p.vals.H_chip = data.via_height(idx);
        
        p.setup.Q0 = global_Q0; 
        p.setup.N  = array_N;
        p.mesh     = mesh_cfg; 
        p.db       = db;
        
        raw_mat = data.Material_Type(idx);
        if iscell(raw_mat), p.mat_type = string(raw_mat{1}); else, p.mat_type = string(raw_mat); end
        
        save_path = fullfile(pwd, modelDir, sprintf('Model_Row_%d.mph', idx));

        try
            res = run_simulation_core(p, save_path);
            
            % 记录结果
            data.T_Max_K(idx)       = res.T_hotspot;
            data.T_Avg_K(idx)       = res.T_avg_vol;
            data.T_Top_Surf_K(idx)  = res.T_top;
            data.T_Bot_Surf_K(idx)  = res.T_bot;
            data.T_Mid_Vol_K(idx)   = res.T_mid_vol; 
            data.T_Mid_Int_K(idx)   = res.T_mid_int; 
            
            data.Stress_Mises_Glob(idx) = res.S_mises_glob;
            data.Stress_Z_Glob(idx)     = res.S_z_glob;     
            data.Stress_1st_Glob(idx)   = res.S_1st_glob;   
            data.Stress_Cu_Mises(idx)   = res.S_cu_mises;
            data.Stress_Ox_Mises(idx)   = res.S_ox_mises;
            
            data.Power_In_W(idx)     = res.Power_in;
            data.Power_Out_W(idx)    = res.Power_out;
            data.Error_Balance_Pct(idx) = res.Error_pct; 
            data.R_th_K_W(idx)       = res.R_th;         
            
            data.Status(idx) = "Success";
            
            fprintf('✅ Row %d 完成 (%.1fs) | T_top: %.2f K | Mises: %.1f MPa\n', ...
                    idx, toc(t_start), res.T_top, res.S_mises_glob);
            
        catch e
            data.Status(idx) = "Error: " + string(e.message);
            fprintf('❌ Row %d 失败: %s\n', idx, e.message);
        end
        writetable(data, csv_output);
        drawnow;
    end
    fprintf('🎉 全部完成。\n');
end

function data = load_data_robust(f_in, f_out)
    data = readtable(f_in);
    vars = data.Properties.VariableNames;
    for i = 1:length(vars)
        if iscell(data.(vars{i})), try data.(vars{i}) = string(data.(vars{i})); catch, end; end
    end
    
    new_cols = {'Material_Type', 'T_Max_K', 'T_Avg_K', 'T_Top_Surf_K', 'T_Bot_Surf_K', ...
                'T_Mid_Vol_K', 'T_Mid_Int_K', 'Stress_Mises_Glob', 'Stress_Z_Glob', ...
                'Stress_1st_Glob', 'Stress_Cu_Mises', 'Stress_Ox_Mises', ...
                'Power_In_W', 'Power_Out_W', 'Error_Balance_Pct', 'R_th_K_W', 'Status'};
    
    for i = 1:length(new_cols)
        col = new_cols{i};
        if ~ismember(col, data.Properties.VariableNames)
            if strcmp(col,'Status') || strcmp(col,'Material_Type')
                data.(col) = repmat("", height(data), 1); 
            else
                data.(col) = nan(height(data), 1);
            end
        end
    end
    
    if isfile(f_out)
        try old = readtable(f_out); 
            n = min(height(old), height(data));
            common_cols = intersect(data.Properties.VariableNames, old.Properties.VariableNames);
            for i=1:length(common_cols), c = common_cols{i}; data.(c)(1:n) = old.(c)(1:n); end
        catch, end
    end
end