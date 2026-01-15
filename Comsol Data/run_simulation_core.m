function result_struct = run_simulation_core(p_struct, save_path)
    import com.comsol.model.*
    import com.comsol.model.util.*

    % --- 0. 初始化 ---
    val_pitch     = 40; 
    val_W_chip    = (p_struct.setup.N+1 ) * val_pitch; 
    
    t_stamp = char(java.util.UUID.randomUUID.toString);
    modelTag = ['M_', strrep(t_stamp, '-', '_')];
    model = ModelUtil.create(modelTag);
    
    try
        model.component.create('comp1', true);
        geom = model.geom.create('geom1', 3);
        geom.lengthUnit('um'); 

        % 1. 参数设置
        v = p_struct.vals;
        fnames = fieldnames(v);
        for i = 1:length(fnames), model.param.set(fnames{i}, sprintf('%.6e[um]', v.(fnames{i}))); end
        
        model.param.set('Q0', sprintf('%.2e[W/m^2]', p_struct.setup.Q0));
        model.param.set('N_arr', num2str(p_struct.setup.N));
        model.param.set('W_chip', sprintf('%.2f[um]', val_W_chip));
        model.param.set('pitch', sprintf('%.2f[um]', val_pitch));
        model.param.set('H_buffer', '20[um]'); 
        model.param.set('H_chip_val', sprintf('%.6e[um]', v.H_chip));
        model.param.set('T_bottom', '293.15[K]');

        % 2. 几何构建
        geom.feature.create('blk_bot','Block').set('size',{'W_chip','W_chip','H_buffer'}).set('pos',{'-W_chip/2','-W_chip/2','-H_buffer'});
        geom.feature.create('blk_mid','Block').set('size',{'W_chip','W_chip','H_chip_val'}).set('pos',{'-W_chip/2','-W_chip/2','0'});
        geom.feature.create('blk_top','Block').set('size',{'W_chip','W_chip','H_buffer'}).set('pos',{'-W_chip/2','-W_chip/2','H_chip_val'});

        exprs = {'r_cu', 'r_cu+t_sio2', 'r_cu+t_sio2+r_D', 'r_cu+t_sio2+r_D+t_sio2', ...
                 'r_cu+t_sio2+r_D+t_sio2+r_cu_1', 'r_cu+t_sio2+r_D+t_sio2+r_cu_1+t_sio2'};
        objs = {};
        for i=1:length(exprs)
            t_cyl = ['c_layer_', num2str(i)];
            c = geom.feature.create(t_cyl, 'Cylinder');
            c.set('r', exprs{i}); c.set('h', 'H_chip_val'); 
            c.set('pos', {'-(N_arr-1)*pitch/2', '-(N_arr-1)*pitch/2', '0'}); 
            
            sel_tag = ['sel_raw_', num2str(i)];
            geom.selection.create(sel_tag, 'CumulativeSelection');
            c.set('contributeto', sel_tag);
            objs{end+1} = t_cyl;
        end
        
        arr = geom.feature.create('arr', 'Array');
        arr.selection('input').set(objs);
        arr.set('fullsize',{'N_arr','N_arr','1'});
        arr.set('displ',{'pitch','pitch','0'});
        
        fin = geom.feature('fin'); fin.set('action', 'union'); fin.set('repairtol', '1e-7[um]'); 
        geom.run; 

        % 3. 选择集 (保持不变)
        for i = 1:6
            clean_tag = ['SEL_CLEAN_', num2str(i)];
            raw_sel_name = ['geom1_sel_raw_', num2str(i), '_dom'];
            if i == 1
                sel_node = model.selection.create(clean_tag, 'Union');
                sel_node.set('input', {raw_sel_name});
            else
                prev_sel_name = ['geom1_sel_raw_', num2str(i-1), '_dom'];
                sel_node = model.selection.create(clean_tag, 'Difference');
                sel_node.set('add', {raw_sel_name});
                sel_node.set('subtract', {prev_sel_name});
            end
        end

        model.selection.create('DOM_MID', 'Box').set('entitydim', 3).set('zmin', '0.1').set('zmax', 'H_chip_val-0.1');
        model.selection.create('DOM_TOP', 'Box').set('entitydim', 3).set('zmin', 'H_chip_val+0.1');
        model.selection.create('DOM_BOT', 'Box').set('entitydim', 3).set('zmax', '-0.1');
        
        model.selection.create('FACE_TOP_EXT', 'Box').set('entitydim', 2).set('zmin', 'H_chip_val+H_buffer-0.1');
        model.selection.create('FACE_BOT_EXT', 'Box').set('entitydim', 2).set('zmax', '-H_buffer+0.1');
        model.selection.create('FACE_MID_BOT', 'Box').set('entitydim', 2).set('zmin', '-0.1').set('zmax', '0.1');
        
        msrc = model.selection.create('FACE_MESH_SRC', 'Box');
        msrc.set('entitydim', 2); msrc.set('zmin', 'H_chip_val-1'); msrc.set('zmax', 'H_chip_val+1');
        msrc.set('xmin','-99999'); msrc.set('xmax','99999'); msrc.set('ymin','-99999'); msrc.set('ymax','99999');
        msrc.set('condition', 'inside');      

        % 4. 材料 (保持不变)
        assign_mat(model, 'Mat_Si_Base', 'Si', 'all', p_struct.db); 
        assign_mat(model, 'M_L1_Cu',     'Cu',              'SEL_CLEAN_1', p_struct.db);
        assign_mat(model, 'M_L2_Ox',     'SiO2',            'SEL_CLEAN_2', p_struct.db);
        assign_mat(model, 'M_L3_Fill',   p_struct.mat_type, 'SEL_CLEAN_3', p_struct.db);
        assign_mat(model, 'M_L4_Ox',     'SiO2',            'SEL_CLEAN_4', p_struct.db);
        assign_mat(model, 'M_L5_Shield', 'Cu',              'SEL_CLEAN_5', p_struct.db);
        assign_mat(model, 'M_L6_Ox',     'SiO2',            'SEL_CLEAN_6', p_struct.db);

        % 5. 物理场 (保持不变)
        ht = model.physics.create('ht', 'HeatTransfer', 'geom1');
        ht.create('hf1','HeatFluxBoundary',2).selection.named('FACE_TOP_EXT'); ht.feature('hf1').set('q0','Q0');
        ht.create('t1','TemperatureBoundary',2).selection.named('FACE_BOT_EXT'); ht.feature('t1').set('T0', 'T_bottom');
        
        solid = model.physics.create('solid','SolidMechanics','geom1');
        solid.selection.named('DOM_MID'); 
        solid.create('roll1', 'Roller', 2).selection.named('FACE_MID_BOT');
        
        te = model.component('comp1').multiphysics.create('te1','ThermalExpansion');
        te.selection.named('DOM_MID'); 
        te.set('Tref', 'T_bottom');

        % --- 6. 网格构建 (【修改】保留扫掠，但移除自定义尺寸控制) ---
        mesh = model.mesh.create('m1', 'geom1');
        
        % 1. 设置全局网格大小 (1=极细 ... 5=正常 ... 9=极粗)
        % 这将自动决定 FreeTri 里的网格密度
        mesh.feature('size').set('hauto', p_struct.mesh.h_auto_level); 
        
        % 2. 在源面建立自由三角形网格
        % 【关键修改】不再添加 'Size' 节点去限制 min_feature，直接由全局 hauto 控制
        ftri = mesh.create('ftri1', 'FreeTri'); 
        ftri.selection.named('FACE_MESH_SRC'); 
        
        % 3. 中间层扫掠 (保留)
        sw_mid = mesh.create('sw_mid', 'Sweep'); 
        sw_mid.selection.named('DOM_MID');
        sw_mid.selection('sourceface').named('FACE_MESH_SRC'); 
        
        dis_mid = sw_mid.create('dis1', 'Distribution');
        dis_mid.set('type', 'predefined');  
        dis_mid.set('numelem', num2str(p_struct.mesh.layers_tsv)); 
        dis_mid.set('elemratio', '2');      
        dis_mid.set('symmetric', 'on');     
        
        % 4. 缓冲层扫掠 (保留)
        sw_top = mesh.create('sw_top', 'Sweep'); sw_top.selection.named('DOM_TOP'); sw_top.selection('sourceface').named('FACE_MESH_SRC');
        sw_top.create('dis2', 'Distribution').set('numelem', num2str(p_struct.mesh.layers_buffer)); 
        
        sw_bot = mesh.create('sw_bot', 'Sweep'); sw_bot.selection.named('DOM_BOT');
        sw_bot.create('dis3', 'Distribution').set('numelem', num2str(p_struct.mesh.layers_buffer)); 

        % 7. 求解
        model.study.create('std1').create('stat','Stationary');
        model.sol.create('s1').createAutoSequence('std1');
        mesh.run; model.sol('s1').runAll; 

        % 8. 结果提取 (保持不变)
        nm = model.result.numerical;
        
        int_top = nm.create('int_flux_in', 'IntSurface'); int_top.selection.named('FACE_TOP_EXT'); int_top.set('expr', 'ht.ntflux');
        int_bot = nm.create('int_flux_out', 'IntSurface'); int_bot.selection.named('FACE_BOT_EXT'); int_bot.set('expr', 'ht.ntflux'); 
        
        av_glob = nm.create('av_T_glob', 'AvVolume'); av_glob.selection.all; av_glob.set('expr', 'T');
        av_mid  = nm.create('av_T_mid_vol', 'AvVolume'); av_mid.selection.named('DOM_MID'); av_mid.set('expr', 'T');
        av_int  = nm.create('av_T_mid_int', 'AvSurface'); av_int.selection.named('FACE_MESH_SRC'); av_int.set('expr', 'T');
        av_top  = nm.create('av_T_top', 'AvSurface'); av_top.selection.named('FACE_TOP_EXT'); av_top.set('expr', 'T');
        av_bot  = nm.create('av_T_bot', 'AvSurface'); av_bot.selection.named('FACE_BOT_EXT'); av_bot.set('expr', 'T');
        max_T   = nm.create('max_T_glob', 'MaxVolume'); max_T.selection.all; max_T.set('expr', 'T');

        max_S_mises = nm.create('max_S_mises', 'MaxVolume'); max_S_mises.selection.named('DOM_MID'); max_S_mises.set('expr', 'solid.mises');
        max_S_z     = nm.create('max_S_z', 'MaxVolume'); max_S_z.selection.named('DOM_MID'); max_S_z.set('expr', 'abs(solid.sz)'); 
        max_S_1     = nm.create('max_S_1st', 'MaxVolume'); max_S_1.selection.named('DOM_MID'); max_S_1.set('expr', 'solid.sp1');

        max_S_cu = get_stress_node(nm, 'max_S_Cu', 'SEL_CLEAN_1', 'solid.mises');
        max_S_ox = get_stress_node(nm, 'max_S_Ox', 'SEL_CLEAN_2', 'solid.mises');

        P_in  = abs(int_top.getReal()); P_out = abs(int_bot.getReal());
        
        result_struct.T_avg_vol = av_glob.getReal(); 
        result_struct.T_top     = av_top.getReal();
        result_struct.T_bot     = av_bot.getReal();
        result_struct.T_mid_vol = av_mid.getReal(); 
        result_struct.T_mid_int = av_int.getReal(); 
        result_struct.T_hotspot = max_T.getReal();
        
        result_struct.S_mises_glob = max_S_mises.getReal() / 1e6;
        result_struct.S_z_glob     = max_S_z.getReal()     / 1e6;
        result_struct.S_1st_glob   = max_S_1.getReal()     / 1e6;
        result_struct.S_cu_mises   = max_S_cu.getReal()    / 1e6;
        result_struct.S_ox_mises   = max_S_ox.getReal()    / 1e6;
        
        result_struct.Power_in  = P_in;
        result_struct.Power_out = P_out;
        result_struct.Error_pct = (abs(P_in - P_out) / P_in) * 100;
        result_struct.R_th = (result_struct.T_top - result_struct.T_bot) / P_in;

        % 9. 保存模型
        if strlength(save_path)>0, try model.save(char(save_path)); catch, end; end

    catch e
        if ~isempty(model), try model.save(fullfile(pwd, ['DEBUG_' t_stamp '.mph'])); catch, end; ModelUtil.remove(modelTag); end
        rethrow(e);
    end
    if ~isempty(model), ModelUtil.remove(modelTag); end
end

function op = get_stress_node(nm, label, sel_tag, expr)
    try
        op = nm.create(label, 'MaxVolume'); op.label(label);
        try op.selection.named(sel_tag); catch, try op.selection.named(['geom1_' sel_tag]); catch, end; end
        op.set('expr', expr);
    catch, op = []; end
end

function assign_mat(model, tag, type, sel, db)
    try model.material.remove(tag); catch, end
    mat = model.material.create(tag, 'Common');
    if strcmp(sel,'all'), mat.selection.all; else, try mat.selection.named(sel); catch, end; end
    set_mat_props(mat, db, type);
end

function set_mat_props(mat, db, type)
    if isfield(db, type), v=db.(type); else, v=db.Cu; end
    keys={'thermalconductivity','density','heatcapacity','youngsmodulus','poissonsratio','thermalexpansioncoefficient'};
    for i=1:6, mat.propertyGroup('def').set(keys{i}, num2str(v(i))); end
end