function gpu_tpot_analyzer()
    % MAIN FUNCTION - Works with text input files
    clc; 
    fprintf('=== GPU TPOT Analyzer ===\n');
    
    % 1. Get input file
    input_file = input('Enter input filename (e.g., layer10.txt): ', 's');
    if isempty(input_file)
        input_file = 'layer10.txt'; % Default filename
    end
    
    % 2. Process input data
    [tpot_matrix, success, num_layers] = process_input_file(input_file);
    if ~success, return; end
    
    fprintf('Detected %d layers in input file\n', num_layers);
    
    % 3. Calculate TPOT and paths
    [optimal_paths, tpot_values] = calculate_tpot_and_paths(tpot_matrix, num_layers);
    
    % 4. Find GHSPA (minimum TPOT)
    [ghspa_tpot, best_gpu] = min(tpot_values);
    ghspa_path = optimal_paths{best_gpu};
    
    % 5. Export to Excel
    output_file = sprintf('gpu_tpot_results_%dlayers.xlsx', num_layers);
    export_to_excel(tpot_matrix, optimal_paths, tpot_values, ghspa_tpot, ghspa_path, output_file, num_layers);
    
    fprintf('\n=== GHSPA Results ===\n');
    fprintf('Minimum TPOT: %d\nPath: %s\n', ghspa_tpot, path_to_string(ghspa_path));
    fprintf('\nResults saved to: %s\n', output_file);
    winopen(output_file);
end

%% Data Processing Function (MUST be included)
function [tpot_matrix, success, num_layers] = process_input_file(input_file)
    success = false;
    num_layers = 0;
    try
        if ~exist(input_file, 'file')
            error('File not found: %s', input_file);
        end
        
        % Read text file line by line
        fid = fopen(input_file, 'r');
        raw_data = textscan(fid, '%s', 'Delimiter', '\n');
        fclose(fid);
        
        num_layers = numel(raw_data{1});
        tpot_matrix = zeros(num_layers, 5); % 5 GPUs
        
        for i = 1:num_layers
            line = strrep(raw_data{1}{i}, 'o', '0'); % Handle 'o' as zero
            nums = sscanf(line, '%f')';
            if length(nums) ~= 5
                error('Line %d must contain exactly 5 values', i);
            end
            tpot_matrix(i,:) = nums;
        end
        success = true;
    catch ME
        fprintf('Error: %s\n', ME.message);
        tpot_matrix = [];
    end
end

%% TPOT Calculation Function (MUST be included)
function [optimal_paths, tpot_values] = calculate_tpot_and_paths(tpot_matrix, num_layers)
    comm_latency = 2; % Communication cost
    tpot_values = zeros(1, 5);
    optimal_paths = cell(1, 5);
    
    for start_gpu = 1:5
        path = start_gpu;
        total_tpot = tpot_matrix(1, start_gpu);
        
        for layer = 2:num_layers
            current_gpu = path(end);
            costs = tpot_matrix(layer,:) + (comm_latency*(1:5 ~= current_gpu));
            [~, next_gpu] = min(costs);
            path = [path, next_gpu];
            total_tpot = total_tpot + costs(next_gpu);
        end
        
        optimal_paths{start_gpu} = path;
        tpot_values(start_gpu) = total_tpot;
    end
end

%% Excel Export Function (MUST be included)
function export_to_excel(tpot_matrix, optimal_paths, tpot_values, ghspa_tpot, ghspa_path, output_file, num_layers)
    % 1. Raw Data Sheet
    headers = {'Layer', 'GPU1', 'GPU2', 'GPU3', 'GPU4', 'GPU5'};
    layer_labels = arrayfun(@(x) sprintf('Layer %d', x), 1:num_layers, 'UniformOutput', false)';
    raw_data = [headers; [layer_labels, num2cell(tpot_matrix)]];
    writecell(raw_data, output_file, 'Sheet', 'Raw Data');
    
    % 2. Results Sheet with GHSPA
    results = {'GPU', 'Total TPOT', 'Optimal Path'};
    for gpu = 1:5
        path_str = path_to_string(optimal_paths{gpu});
        results = [results; {sprintf('GPU%d',gpu), tpot_values(gpu), path_str}];
    end
    
    % Add GHSPA row
    results = [results; {'GHSPA', ghspa_tpot, path_to_string(ghspa_path)}];
    
    writecell(results, output_file, 'Sheet', 'Results');
end

%% Helper Function (MUST be included)
function str = path_to_string(path)
    str = strjoin(arrayfun(@(x) sprintf('GPU%d',x), path, 'UniformOutput', false), ' → ');
end