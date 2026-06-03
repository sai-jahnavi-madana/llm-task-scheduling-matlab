clc;
clear;
close all;

tpot = [
    1, 10, 1;
    10, 1, 10;
    1, 10, 1
];

layer_mem = [2; 4; 2];

gpu_mem_initial = [5, 6, 5];

comm = [
    0, 2, 0;
    2, 0, 2;
    0, 2, 0
];

num_layers = size(tpot, 1);
num_gpus = size(tpot, 2);

dp_cost = inf(num_layers, num_gpus);
dp_path = cell(num_layers, num_gpus);
dp_rem_mem = cell(num_layers, num_gpus);

for g = 1:num_gpus
    if layer_mem(1) <= gpu_mem_initial(g)
        dp_cost(1, g) = tpot(1, g);
        dp_path{1, g} = g;
        dp_rem_mem{1, g} = gpu_mem_initial;
        dp_rem_mem{1, g}(g) = dp_rem_mem{1, g}(g) - layer_mem(1);
    end
end

for l = 2:num_layers
    current_layer_mem = layer_mem(l);
    for current_gpu = 1:num_gpus
        
        if current_layer_mem > gpu_mem_initial(current_gpu)
            continue;
        end

        for prev_gpu = 1:num_gpus
            if dp_cost(l-1, prev_gpu) ~= inf
                
                prev_rem_mem = dp_rem_mem{l-1, prev_gpu};
                
                if current_layer_mem <= prev_rem_mem(current_gpu)
                    
                    transition_cost = tpot(l, current_gpu);
                    if current_gpu ~= prev_gpu
                        transition_cost = transition_cost + comm(prev_gpu, current_gpu);
                    end
                    
                    total_current_cost = dp_cost(l-1, prev_gpu) + transition_cost;
                    
                    if total_current_cost < dp_cost(l, current_gpu)
                        dp_cost(l, current_gpu) = total_current_cost;
                        
                        dp_path{l, current_gpu} = [dp_path{l-1, prev_gpu}, current_gpu];
                        
                        new_rem_mem = prev_rem_mem;
                        new_rem_mem(current_gpu) = new_rem_mem(current_gpu) - current_layer_mem;
                        dp_rem_mem{l, current_gpu} = new_rem_mem;
                    end
                end
            end
        end
    end
end

[min_total_cost, last_gpu_idx] = min(dp_cost(num_layers, :));

if min_total_cost == inf
    fprintf('No feasible path found under given constraints.\n');
else
    optimal_path = dp_path{num_layers, last_gpu_idx};
    
    fprintf('Optimal Path : ');
    fprintf('GPU%d ', optimal_path);
    fprintf('\n');
    fprintf('DPTSA Value: %.2f ms\n', min_total_cost);
    
    final_mem_check = gpu_mem_initial;
    for l_idx = 1:num_layers
        assigned_gpu = optimal_path(l_idx);
        final_mem_check(assigned_gpu) = final_mem_check(assigned_gpu) - layer_mem(l_idx);
    end
end