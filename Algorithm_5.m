function greedy_ghspa_scheduler()
    time_matrix = dlmread('time100gpu5.txt'); 
    cost_matrix = dlmread('cost100gpu5.txt'); 
    
    comm_matrix = [
        0, 3, 2, 4, 1;  
        4, 0, 1, 2, 3;  
        3, 2, 0, 5, 2;  
        5, 1, 4, 0, 3;  
        2, 4, 3, 2, 0   
    ];
    
    layer_mem = 3 * ones(1, size(time_matrix, 1));
    gpu_mem = [7, 8, 6, 7, 10];
    lambda = 0.25;
    
    num_layers = size(time_matrix, 1);
    num_gpus = size(time_matrix, 2);
    path = zeros(1, num_layers);
    total_time = 0;
    total_cost = 0;
    total_comm = 0;
    
    for layer = 1:num_layers
        best_gpu = 0;
        min_cost = Inf;
        current_comm = 0;
        
        for gpu = 1:num_gpus
            if gpu_mem(gpu) >= layer_mem(layer) 
                if layer == 1
                    comm = 0;
                else
                    if path(layer-1) > 0 && path(layer-1) <= size(comm_matrix, 1) && ...
                       gpu > 0 && gpu <= size(comm_matrix, 2)
                        comm = comm_matrix(path(layer-1), gpu);
                    else
                        warning('Invalid GPU index for communication matrix at layer %d: prev_gpu=%d, current_gpu=%d', layer, path(layer-1), gpu);
                        comm = Inf; 
                    end
                end
                
                cost = lambda * (time_matrix(layer, gpu) + comm) + ...
                      (1 - lambda) * cost_matrix(layer, gpu);
                
                if cost < min_cost
                    min_cost = cost;
                    best_gpu = gpu;
                    current_comm = comm;
                end
            end
        end
        
        if best_gpu == 0
            error('No GPU has sufficient memory for layer %d (required: %gGB)', layer, layer_mem(layer));
        end
        
        path(layer) = best_gpu;
        
        total_time = total_time + time_matrix(layer, best_gpu);
        total_cost = total_cost + cost_matrix(layer, best_gpu);
        total_comm = total_comm + current_comm;
    end
    
    ghspa_value = lambda*(total_time + total_comm) + (1-lambda)*total_cost;

    path_str_cells = cell(1, num_layers);
    for i = 1:num_layers
        path_str_cells{i} = sprintf('gpu%d', path(i));
    end
    optimal_path_display = strjoin(path_str_cells, '->');

    output_data = {'Optimal Path', optimal_path_display; ...
                   'GHSPA Value', ghspa_value};

    fprintf('Optimal Path: %s\n', optimal_path_display);
    fprintf('GHSPA Value: %.2f\n', ghspa_value);
end