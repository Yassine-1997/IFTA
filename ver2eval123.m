% Initialize common parameters
initial_energy = 1872; % Initial energy in Joules for each node
E_tx = 50e-5; % Energy consumption coefficient for data transmission (J/bit)
E_rx = 50e-5; % Energy consumption coefficient for data reception (J/bit)
num_iterations = 100; % Number of iterations for Tabu Search

% Define network size and sink node
network_size = 100;
sink_node = [250, 250]; % Sink node coordinates

% Vary the total number of nodes
node_counts = 200:100:900;
total_energy_consumption = zeros(size(node_counts));

% Loop through each number of nodes
for count_idx = 1:length(node_counts)
    numNodes = node_counts(count_idx);
    nodes = [rand(numNodes, 1) * network_size, rand(numNodes, 1) * network_size];
    energy_levels = initial_energy * ones(numNodes, 1);
    energy_consumption = zeros(numNodes, 1); % Initialize energy consumption array

    % Prompt user for the optimal number of clusters (k)
    optimal_k = input(['Enter the optimal number of clusters (k) for ' num2str(numNodes) ' nodes: ']);

    % Perform K-means clustering
    [idx, C] = kmeans(nodes, optimal_k);

    % Initialize the best solution for each cluster
    best_solutions = cell(1, optimal_k);
    best_costs = inf * ones(1, optimal_k);
    best_cost_iter = zeros(optimal_k, num_iterations);

    % Define distance function
    distance = @(a, b) sqrt((a(:, 1) - b(1)).^2 + (a(:, 2) - b(2)).^2);

    % Define cost function
    cost_function = @(x, ch_index) sum(sqrt((x(1:end-1, 1) - x(2:end, 1)).^2 + (x(1:end-1, 2) - x(2:end, 2)).^2)) + ...
                                     sum(sqrt((x(:, 1) - sink_node(1)).^2 + (x(:, 2) - sink_node(2)).^2)) + ...
                                     E_tx * sum(distance(x, x(ch_index, :)));

    % Loop through each cluster
    for i = 1:optimal_k
        % Get the nodes in the cluster
        cluster_nodes = nodes(idx == i, :);
        num_cluster_nodes = size(cluster_nodes, 1);

        % Initialize the Tabu Search for this cluster
        tabu_list = zeros(num_cluster_nodes, num_iterations);

        % Run the Tabu Search
        for iter = 1:num_iterations
            % Initialize the current solution
            current_solution = randperm(num_cluster_nodes);
            current_cost = cost_function(cluster_nodes(current_solution, :), 1);

            % Run the Tabu Search iterations
            for j = 1:num_cluster_nodes
                % Generate the neighborhood
                neighborhood = current_solution;
                neighborhood(j) = current_solution(mod(j+1, num_cluster_nodes)+1);

                % Evaluate the cost of the neighborhood
                neighborhood_cost = cost_function(cluster_nodes(neighborhood, :), neighborhood(1));

                % Calculate energy consumption for data transmission
                energy_consumption_tx = E_tx * distance(cluster_nodes(current_solution(j), :), cluster_nodes(current_solution(mod(j+1, num_cluster_nodes)+1), :));

                % Calculate energy consumption for data reception
                energy_consumption_rx = E_rx * distance(cluster_nodes(current_solution(j), :), cluster_nodes(current_solution(mod(j+1, num_cluster_nodes)+1), :));

                % Update energy levels of nodes
                energy_levels(current_solution(j)) = energy_levels(current_solution(j)) - energy_consumption_tx - energy_consumption_rx;
                energy_consumption(current_solution(j)) = energy_consumption(current_solution(j)) + energy_consumption_tx + energy_consumption_rx;

                % Check if the neighborhood cost is better than the current cost
                if neighborhood_cost < current_cost
                    current_solution = neighborhood;
                    current_cost = neighborhood_cost;
                end
            end

            % Update the best solution if the current cost is better
            if current_cost < best_costs(i)
                best_solutions{i} = cluster_nodes(current_solution(1), :);
                best_costs(i) = current_cost;
            end

            % Update the tabu list
            tabu_list(:, iter) = current_solution';

            % Update the best_cost_iter array
            if iter > 1
                best_cost_iter(i, iter) = min(best_cost_iter(i, iter - 1), current_cost);
            else
                best_cost_iter(i, iter) = current_cost;
            end
        end
    end

    % Calculate total energy consumption by all nodes
    total_energy_consumption(count_idx) = sum(energy_consumption);
end

% Plot total energy consumption vs. number of nodes
figure;
plot(node_counts, total_energy_consumption, '-o');
xlabel('Total Number of Nodes');
ylabel('Total Energy Consumption (J)');
title('Total Energy Consumption vs. Number of Nodes');
grid on;
