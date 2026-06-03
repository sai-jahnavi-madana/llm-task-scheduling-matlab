time_matrix = [     
    1 8 2 5 1;     
    6 1 7 1 6;     
    2 5 1 4 1;     
    1 3 6 1 5;     
    4 1 3 5 1 
];
cost_matrix = [     
    5 7 3 2 4;     
    6 4 3 5 1;     
    2 6 7 5 1;     
    3 2 4 6 5;     
    5 4 3 1 6 
];
lambda = 0.5;
[num_layers, num_machines] = size(time_matrix);
assignments = zeros(1, num_layers);
assigned_times = zeros(1, num_layers);
assigned_costs = zeros(1, num_layers);
assigned_m_values = zeros(1, num_layers);

for i = 1:num_layers
    fprintf('Layer %d:\n', i);
  
    for j = 1:num_machines
        fprintf('  Machine %d: Time = %d, Cost = %d\n', j, time_matrix(i,j), cost_matrix(i,j));
    end
    m_values = lambda * time_matrix(i,:) + (1 - lambda) * cost_matrix(i,:);
  
    fprintf('  Combined m values: ');
    fprintf('%.2f ', m_values);
    fprintf('\n');

    [min_m_value, best_machine] = min(m_values);
    assignments(i) = best_machine;
    assigned_times(i) = time_matrix(i, best_machine);
    assigned_costs(i) = cost_matrix(i, best_machine);
    assigned_m_values(i) = min_m_value;
    
    fprintf('  ==> Best Machine: %d with Time = %d, Cost = %d, m-value = %.2f\n\n', ...
        best_machine, assigned_times(i), assigned_costs(i), min_m_value);
end
TPOT = sum(assigned_times);
total_cost = sum(assigned_costs);
total_m = sum(assigned_m_values);

fprintf('Summary:\n');
fprintf('Total Processing Time (TPOT): %d\n', TPOT);
fprintf('Total Cost: %d\n', total_cost);

fprintf('Layer assignments (Layer -> Best Machine):\n');
for i = 1:num_layers
    fprintf('  Layer %d -> Machine %d\n', i, assignments(i));
end
