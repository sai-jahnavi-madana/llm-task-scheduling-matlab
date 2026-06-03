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
if layer_mem(1) <= gpu_mem_initial(1)
    dp_cost(1, 1) = tpot(1, 1);
    dp_path{1, 1} = [1]; % Path starts with GPU 1
    dp_rem_mem{1, 1} = gpu_mem_initial;
    dp_rem_mem{1, 1}(1) = dp_rem_mem{1, 1}(1) - layer_mem(1);
end
for l = 2:num_layers-1 
    current_layer_mem = layer_mem(l);
    for current_gpu = 1:num_gpus
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

min_cost_to_final_gpu1 = inf;
optimal_final_path = [];

for prev_gpu = 1:num_gpus
    if dp_cost(num_layers-1, prev_gpu) ~= inf
        
        prev_rem_mem = dp_rem_mem{num_layers-1, prev_gpu};
        
        if layer_mem(num_layers) <= prev_rem_mem(1)
            
            transition_cost = tpot(num_layers, 1); 
          
            if 1 ~= prev_gpu
                transition_cost = transition_cost + comm(prev_gpu, 1);
            end
            
            current_final_cost = dp_cost(num_layers-1, prev_gpu) + transition_cost;
          
            if current_final_cost < min_cost_to_final_gpu1
                min_cost_to_final_gpu1 = current_final_cost;
                optimal_final_path = [dp_path{num_layers-1, prev_gpu}, 1]; 
            end
        end
    end
end


fprintf('--- Modified DPTSA---\n');
if min_cost_to_final_gpu1 == inf
    fprintf('No feasible path found under given constraints.\n');
else
    fprintf('Optimal Path : ');
    fprintf('GPU%d ', optimal_final_path);
    fprintf('\n');
    fprintf('Modified DPTSA: %.2f ms\n', min_cost_to_final_gpu1);
end