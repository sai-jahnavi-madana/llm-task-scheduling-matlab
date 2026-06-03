clc;
clear;
close all;

%% --- Common Input Data ---
time_matrix = [1    8    2    5    1;
               6    1    7    1    6;
               2    5    1    4    1;
               1    3    6    1    5;
               4    1    3    5    1];

cost_matrix = [5    7    3    2    4;
               6    4    3    5    1;
               2    6    7    5    1;
               3    2    4    6    5;
               5    4    3    1    6];

comm_matrix = [0   3   1   2   1;
               3   0   2   1   3;
               1   2   0   2   1;
               2   1   2   0   3;
               1   3   1   3   0];

lambda = 0.75;

layer_mem = [3, 5, 2, 4, 3];
gpu_mem = [7, 8, 6, 7, 10];

%% --- Run GHSPA Algorithm ---
disp('--- Running GHSPA (Greedy Heuristic - Layer-by-Layer Assignment) ---');
[ghspa_value, ghspa_path, ghspa_total_time, ghspa_total_cost] = ...
    ghspa_scheduler(time_matrix, cost_matrix, layer_mem, gpu_mem, comm_matrix, lambda);
fprintf('Greedy Optimal Path (GHSPA): ');
fprintf('GPU%d ', ghspa_path(1:end-1));
fprintf('GPU%d\n', ghspa_path(end));
fprintf('GHSPA Value (λ=%.2f): %.2f\n', lambda, ghspa_value);


%% --- Run IGHSPA Algorithm ---
disp('--- Running IGHSPA (Improved Greedy Heuristic - Single GPU Assignment) ---');
[ighspa_value, ighspa_path, ighspa_total_time, ighspa_total_cost] = ...
    ighspa_scheduler(time_matrix, cost_matrix, layer_mem, gpu_mem, comm_matrix, lambda);
fprintf('Optimal GPU Path (IGHSPA - Single GPU): ');
fprintf('GPU%d ', ighspa_path(1:end-1));
fprintf('GPU%d\n', ighspa_path(end));
fprintf('IGHSPA Value (λ=%.2f): %.2f\n', lambda, ighspa_value);

%% --- Run DPTSA Algorithm ---
disp('--- Running DPTSA (Dynamic Programming - Exact) ---');
[dptsa_value, dptsa_path, dptsa_total_time, dptsa_total_cost] = ...
    dptsa_time_cost_single_comm(time_matrix, cost_matrix, layer_mem, gpu_mem, comm_matrix, lambda);
if isnan(dptsa_value)
    disp('DPTSA: No feasible path found considering memory constraints.');
else
    disp(['Optimal GPU Path (DPTSA): ', sprintf('GPU%d ', dptsa_path)]);
    disp(['DPTSA Value (lambda=', num2str(lambda), '): ', num2str(dptsa_value)]);
end
fprintf('\n');

%% --- Local Functions ---

function [ghspa_value, path, total_time_actual, total_cost_actual] = ghspa_scheduler(time_matrix, cost_matrix, layer_mem, gpu_mem, comm_matrix, lambda)
    num_layers = size(time_matrix, 1);
    num_gpus = size(time_matrix, 2);
    path = zeros(1, num_layers);
    remaining_mem = gpu_mem;
    total_time_actual = 0;
    total_cost_actual = 0;

    for layer = 1:num_layers
        best_gpu = 0;
        min_immediate_weighted_cost = Inf;
        current_layer_proc_time_chosen = 0;
        current_layer_proc_cost_chosen = 0;
        current_layer_comm_cost_chosen = 0;

        for gpu = 1:num_gpus
            if layer_mem(layer) <= remaining_mem(gpu)
                comm_cost_candidate = 0;
                if layer > 1
                    if gpu ~= path(layer-1)
                        comm_cost_candidate = comm_matrix(path(layer-1), gpu);
                    end
                end

                immediate_weighted_cost_candidate = ...
                    lambda * (time_matrix(layer, gpu) + comm_cost_candidate) + ...
                    (1 - lambda) * (cost_matrix(layer, gpu) + comm_cost_candidate);

                if immediate_weighted_cost_candidate < min_immediate_weighted_cost
                    min_immediate_weighted_cost = immediate_weighted_cost_candidate;
                    best_gpu = gpu;
                    current_layer_proc_time_chosen = time_matrix(layer, gpu);
                    current_layer_proc_cost_chosen = cost_matrix(layer, gpu);
                    current_layer_comm_cost_chosen = comm_cost_candidate;
                end
            end
        end

        if best_gpu == 0
            error('No feasible GPU found for layer %d (memory constraints or all paths too costly)', layer);
        end

        path(layer) = best_gpu;
        remaining_mem(best_gpu) = remaining_mem(best_gpu) - layer_mem(layer);
        total_time_actual = total_time_actual + current_layer_proc_time_chosen + current_layer_comm_cost_chosen;
        total_cost_actual = total_cost_actual + current_layer_proc_cost_chosen + current_layer_comm_cost_chosen;
    end
    ghspa_value = lambda * total_time_actual + (1 - lambda) * total_cost_actual;
end


function [weighted_value, path, total_time, total_cost] = ighspa_scheduler(time_matrix, cost_matrix, layer_mem, gpu_mem, comm_matrix, lambda)
    num_layers = size(time_matrix, 1);
    num_gpus = size(time_matrix, 2);

    candidate_total_times = zeros(1, num_gpus);
    candidate_total_costs = zeros(1, num_gpus);

    candidate_total_times(:) = Inf;
    candidate_total_costs(:) = Inf;

    for gpu = 1:num_gpus
        if all(layer_mem <= gpu_mem(gpu))
            candidate_total_times(gpu) = sum(time_matrix(:, gpu));
            candidate_total_costs(gpu) = sum(cost_matrix(:, gpu));
        end
    end

    [min_total_time, best_gpu_idx] = min(candidate_total_times);

    if isinf(min_total_time)
        weighted_value = NaN;
        path = NaN(1, num_layers);
        total_time = NaN;
        total_cost = NaN;
        error('IGHSPA: No single GPU can fit all layers for calculation. Consider relaxing memory constraints or using a different scheduler.');
    end

    total_time = min_total_time;
    total_cost = candidate_total_costs(best_gpu_idx);

    path = ones(1, num_layers) * best_gpu_idx;

    weighted_value = lambda * total_time + (1 - lambda) * total_cost;
end

function [min_weighted_value, optimal_path, total_time, total_cost] = dptsa_time_cost_single_comm(time_matrix, cost_matrix, layer_mem, gpu_mem, comm_matrix, lambda)
    [num_layers, num_gpus] = size(time_matrix);

    dp_weighted_value = inf(num_layers, num_gpus);
    dp_prev_gpu = zeros(num_layers, num_gpus);
    dp_actual_time = zeros(num_layers, num_gpus);
    dp_actual_cost = zeros(num_layers, num_gpus);

    for g = 1:num_gpus
        if layer_mem(1) <= gpu_mem(g)
            dp_weighted_value(1, g) = lambda * time_matrix(1, g) + (1 - lambda) * cost_matrix(1, g);
            dp_actual_time(1, g) = time_matrix(1, g);
            dp_actual_cost(1, g) = cost_matrix(1, g);
            dp_prev_gpu(1, g) = -1;
        end
    end

    for l = 2:num_layers
        for g = 1:num_gpus
            if layer_mem(l) > gpu_mem(g)
                continue;
            end

            min_prev_weighted_val = inf;
            best_prev_gpu_for_this_state = 0;
            actual_time_for_this_state = inf;
            actual_cost_for_this_state = inf;

            for prev_g = 1:num_gpus
                if isinf(dp_weighted_value(l-1, prev_g))
                    continue;
                end

                current_layer_proc_time = time_matrix(l, g);
                current_layer_proc_cost = cost_matrix(l, g);

                comm_val = 0;
                if prev_g ~= g
                    comm_val = comm_matrix(prev_g, g);
                end

                path_time = dp_actual_time(l-1, prev_g) + current_layer_proc_time + comm_val;
                path_cost = dp_actual_cost(l-1, prev_g) + current_layer_proc_cost + comm_val;

                current_weighted_value_candidate = lambda * path_time + (1 - lambda) * path_cost;

                if current_weighted_value_candidate < min_prev_weighted_val
                    min_prev_weighted_val = current_weighted_value_candidate;
                    best_prev_gpu_for_this_state = prev_g;
                    actual_time_for_this_state = path_time;
                    actual_cost_for_this_state = path_cost;
                end
            end

            dp_weighted_value(l, g) = min_prev_weighted_val;
            dp_prev_gpu(l, g) = best_prev_gpu_for_this_state;
            dp_actual_time(l, g) = actual_time_for_this_state;
            dp_actual_cost(l, g) = actual_cost_for_this_state;
        end
    end

    [min_weighted_value, last_gpu_idx] = min(dp_weighted_value(end, :));

    if isinf(min_weighted_value)
        optimal_path = NaN(1, num_layers);
        total_time = NaN;
        total_cost = NaN;
        return;
    end

    optimal_path = zeros(1, num_layers);
    optimal_path(end) = last_gpu_idx;

    for l = num_layers:-1:2
        optimal_path(l-1) = dp_prev_gpu(l, optimal_path(l));
    end

    total_time = dp_actual_time(num_layers, last_gpu_idx);
    total_cost = dp_actual_cost(num_layers, last_gpu_idx);
end