function [best_path, ighspa_value] = IghspaTimeCost()
    try
        time_matrix = dlmread('time100gpu5.txt');
        cost_matrix = dlmread('cost100gpu5.txt');
        
        layer_mem =  ones(1, 100);
        gpu_mem = [16, 24, 32, 16, 24];    
        lambda = 0.25;                      
        
        [best_path, ighspa_value] = calculate_ighspa(time_matrix, cost_matrix, lambda, layer_mem, gpu_mem);
        
        fprintf('IGHSPA Value: %.2f\n', ighspa_value);
        fprintf('Optimal Path: ');
        fprintf('gpu%d->', best_path(1:end-1));
        fprintf('gpu%d\n', best_path(end));
        
        if nargout == 0
            clear best_path ighspa_value;
        end
        
    catch ME
        error('Error in IGHSPA calculation: %s', ME.message);
    end
end
function [best_path, ighspa_value] = calculate_ighspa(time_mat, cost_mat, lambda, layer_mem, gpu_mem)
    num_layers = length(layer_mem);
    num_gpus = length(gpu_mem);
    
    best_path = ones(1, num_layers);
    ighspa_value = Inf;
    
    remaining_mem = gpu_mem;
    for layer = 1:num_layers
        valid_gpus = find(remaining_mem >= layer_mem(layer));
        if isempty(valid_gpus)
            error('No valid allocation for layer %d', layer);
        end
        
        m_values = lambda * time_mat(layer, valid_gpus) + (1-lambda) * cost_mat(layer, valid_gpus);
        [~, idx] = min(m_values);
        best_path(layer) = valid_gpus(idx);
        remaining_mem(best_path(layer)) = remaining_mem(best_path(layer)) - layer_mem(layer);
    end
    
    total_time = sum(time_mat(sub2ind(size(time_mat), 1:num_layers, best_path)));
    total_cost = sum(cost_mat(sub2ind(size(cost_mat), 1:num_layers, best_path)));
    ighspa_value = lambda * total_time + (1-lambda) * total_cost;
end