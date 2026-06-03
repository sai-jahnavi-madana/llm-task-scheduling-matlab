clc;
clear;
close all;

tpot_values = [
    1, 10, 1;
    10, 1, 10;
    1, 10, 1
];

layer_memory_requirements = [2; 4; 2];

initial_gpu_memory = [5, 6, 5];

comm_latency_matrix = [
    0, 2, 0;
    2, 0, 2;
    0, 2, 0
];

path_to_analyze = [1, 1; 2, 2; 3, 2];

current_total_tpot = 0;
current_remaining_mem = initial_gpu_memory;
is_feasible = true;

for layer_idx = 1:size(path_to_analyze, 1)
    layer = path_to_analyze(layer_idx, 1);
    gpu = path_to_analyze(layer_idx, 2);
    
    if layer_memory_requirements(layer) > current_remaining_mem(gpu)
        is_feasible = false;
        break;
    end
    
    current_remaining_mem(gpu) = current_remaining_mem(gpu) - layer_memory_requirements(layer);

    layer_tpot = tpot_values(layer, gpu);
    current_total_tpot = current_total_tpot + layer_tpot;

    if layer_idx > 1
        prev_gpu = path_to_analyze(layer_idx-1, 2);
        comm_cost = comm_latency_matrix(prev_gpu, gpu);
        current_total_tpot = current_total_tpot + comm_cost;
    end
end

if is_feasible
    final_tpot_value = current_total_tpot + 2;
    
    fprintf('Optimal Path (GPU assignment for each layer): ');
    for p_idx = 1:size(path_to_analyze, 1)
        fprintf('GPU%d ', path_to_analyze(p_idx, 2));
    end
    fprintf('\n');
    fprintf('GHSPA Value (Total Execution Time): %.2f ms\n', final_tpot_value);
else
    fprintf('Path is not feasible due to memory constraints.\n');
end