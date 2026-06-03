function dptsa()
    clc;
    clear;
    close all;

    time_matrix = dlmread('time100gpu5.txt');
    cost_matrix = dlmread('cost100gpu5.txt');
    comm_matrix = [
        0, 3, 2, 4, 1;
        4, 0, 1, 2, 3;
        3, 2, 0, 5, 2;
        5, 1, 4, 0, 3;
        2, 4, 3, 2, 0
    ];
    layer_mem = ones(1, 100);
    gpu_mem = [16, 24, 32, 16, 24];
    lambda = 0.25;

    tpot = time_matrix;
    comm = comm_matrix;
    gpu_mem_initial = gpu_mem;

    num_layers = size(tpot, 1);
    num_gpus = size(tpot, 2);

    dp_cost = inf(num_layers, num_gpus);
    dp_path = cell(num_layers, num_gpus);
    dp_rem_mem = cell(num_layers, num_gpus);

    for g = 1:num_gpus
        dp_cost(1, g) = lambda * time_matrix(1, g) + (1 - lambda) * cost_matrix(1, g);
        
        dp_path{1, g} = g;
        dp_rem_mem{1, g} = gpu_mem_initial;
        dp_rem_mem{1, g}(g) = dp_rem_mem{1, g}(g) - layer_mem(1);
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
                        
                        exec_time_current = time_matrix(l, current_gpu);
                        
                        resource_cost_current = cost_matrix(l, current_gpu);
                        
                        comm_cost = 0;
                        if current_gpu ~= prev_gpu
                            comm_cost = comm_matrix(prev_gpu, current_gpu);
                        end
                        
                        weighted_transition_cost = lambda * (exec_time_current + comm_cost) + (1 - lambda) * resource_cost_current;
                        
                        total_current_cost = dp_cost(l-1, prev_gpu) + weighted_transition_cost;

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
        fprintf('IDPTSA Value : %.2f\n', min_total_cost);

        final_mem_check = gpu_mem_initial;
        for l_idx = 1:num_layers
            assigned_gpu = optimal_path(l_idx);
            final_mem_check(assigned_gpu) = final_mem_check(assigned_gpu) - layer_mem(l_idx);
        end
    end
end