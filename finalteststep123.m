% Initialize the WSN
numNodes = 70;
nodes = [rand(numNodes, 1) * 500, rand(numNodes, 1) * 500];
sink_node = [250,250]; % Define the sink node
sink_x = 250; % Sink node x-coordinate
sink_y = 250; % Sink node y-coordinate

% Perform K-means clustering
optimal_k = input('Enter the optimal number of clusters (k): ');
[idx, C] = kmeans(nodes, optimal_k);

% Initialize the Tabu Search
num_iterations = 100;
tabu_tenure = randi(num_iterations, numNodes, 1);

% Initialize the energy consumption coefficient
energy_consumption_coefficient = 0.1; % Coefficient to model energy consumption

% Initialize the cost functions
distance = @(a, b) sqrt((a(:, 1) - b(1)).^2 + (a(:, 2) - b(2)).^2);

cost_function = @(x, ch_index) sum(sqrt((x(1:end-1, 1) - x(2:end, 1)).^2 + (x(1:end-1, 2) - x(2:end, 2)).^2)) + ...
                                 sum(sqrt((x(:, 1) - sink_node(1)).^2 + (x(:, 2) - sink_node(2)).^2)) + ...
                                 energy_consumption_coefficient * sum(distance(x, x(ch_index, :)));

% Initialize the best solution for each cluster
best_solutions = cell(1, optimal_k);
best_costs = inf * ones(1, optimal_k);
best_cost_iter = zeros(optimal_k, num_iterations);

% Define colors for marking selected cluster heads
colors = ['r', 'g', 'b', 'm', 'c', 'y'];

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

        % Initialize the tabu list for this iteration
        tabu_list(:, iter) = current_solution';

        % Run the Tabu Search iterations
        for j = 1:num_cluster_nodes
            % Generate the neighborhood
            neighborhood = current_solution;
            neighborhood(j) = current_solution(mod(j+1, num_cluster_nodes)+1);

            % Evaluate the cost of the neighborhood
            neighborhood_cost = cost_function(cluster_nodes(neighborhood, :), neighborhood(1));

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
        tabu_tenure(idx == i) = tabu_tenure(idx == i) - 1;

        % Update the best_cost_iter array
        if iter > 1
            best_cost_iter(i, iter) = min(best_cost_iter(i, iter - 1), current_cost);
        else
            best_cost_iter(i, iter) = current_cost;
        end

    end
end

% Step 4: Construct MST for CHs and Sink node using Kruskal's Algorithm
MST_edges = kruskal_mst(best_solutions, sink_x, sink_y); % Compute MST

% Save the relevant variables to a .mat file
save('wsn_results.mat', 'nodes', 'idx', 'C', 'best_solutions', 'sink_node', 'MST_edges', '-v7.3');
disp('Clustering and network initialization results saved successfully!');



% Plot the WSN with selected cluster heads
figure;
hold on;
for i = 1:optimal_k
    cluster_nodes = nodes(idx == i, :);
    selected_ch = best_solutions{i};

    % Plot cluster nodes
    plot(cluster_nodes(:, 1), cluster_nodes(:, 2), 'o', 'Color', colors(mod(i, length(colors)) + 1), 'MarkerFaceColor', colors(mod(i, length(colors)) + 1));

    % Plot selected cluster head
    plot(selected_ch(1), selected_ch(2), 'x', 'MarkerSize', 12, 'LineWidth', 2, 'Color', colors(mod(i, length(colors)) + 1));

    % Plot connection between cluster nodes and cluster head
    for j = 1:size(cluster_nodes, 1)
        plot([cluster_nodes(j, 1), selected_ch(1)], [cluster_nodes(j, 2), selected_ch(2)], 'k');
    end
end

% Plot sink node
plot(sink_node(1), sink_node(2), 's', 'MarkerSize', 15, 'LineWidth', 2, 'Color', 'k', 'MarkerFaceColor', 'k');
xlabel('X-coordinate');
ylabel('Y-coordinate');
title('WSN with Selected Cluster Heads');
axis equal;
grid on;
legend('Cluster Nodes', 'Cluster Head', 'Connection to Sink', 'Sink Node');

% Plot final network including CHs, Sink, and MST edges
figure;
hold on;

% Plot MST edges
for i = 1:size(MST_edges, 1)
    plot(MST_edges(i, [1, 3]), MST_edges(i, [2, 4]), 'b-', 'LineWidth', 1.5);
end

% Plot CHs
for i = 1:optimal_k
    clusterNodes = nodes(idx == i, :);
    scatter(clusterNodes(:, 1), clusterNodes(:, 2), 'filled');
    scatter(best_solutions{i}(1), best_solutions{i}(2), 100, 'r', 'filled'); % Plot CHs
    % Connect nodes in the cluster to the corresponding CH
    for j = 1:size(clusterNodes, 1)
        plot([clusterNodes(j, 1), best_solutions{i}(1)], [clusterNodes(j, 2), best_solutions{i}(2)], 'k');
    end
end

% Plot Sink node
scatter(sink_node(1), sink_node(2), 100, 'g', 'filled');
text(sink_node(1) + 2, sink_node(2) + 2, 'Sink', 'Color', 'g');

title('Final Network Visualization');
xlabel('X-coordinate');
ylabel('Y-coordinate');
grid on;

function MST_edges = kruskal_mst(best_solutions, sink_x, sink_y)
    numCH = length(best_solutions); % Number of Cluster Heads
    
    % Combine CHs and Sink node
    CH_XY = [vertcat(best_solutions{:}); [sink_x, sink_y]];

    % Compute pairwise distances between CHs and Sink node
    distances = pdist2(CH_XY, CH_XY);

    % Initialize the MST edges
    MST_edges = zeros(numCH+1, 4);
    
    % Initialize the edges matrix
    edges = zeros(numCH*(numCH-1)/2 + numCH, 3);
    k = 1;
    for i = 1:numCH
        for j = i+1:numCH
            edges(k, 1) = i;
            edges(k, 2) = j;
            edges(k, 3) = distances(i, j);
            k = k + 1;
        end
    end
    
    % Add edges from CHs to Sink node
    for i = 1:numCH
        edges(k, 1) = i;
        edges(k, 2) = numCH + 1;
        edges(k, 3) = pdist([CH_XY(i,:); [sink_x, sink_y]]);
        k = k + 1;
    end
    
    % Sort edges by weight
    edges = sortrows(edges, 3);
    
    % Initialize the parents array
    parent = 1:numCH+1;
    
    % Initialize the number of edges added to MST
    edge_count = 0;
    
    % Iterate over all edges in sorted order
    idx = 1;
    while edge_count < numCH
        % Extract the next edge
        u = edges(idx, 1);
        v = edges(idx, 2);
        w = edges(idx, 3);
        idx = idx + 1;
        
        % Find parents of vertices u and v
        set_u = find_parent(parent, u);
        set_v = find_parent(parent, v);
        
        % If including this edge does not form a cycle, include it
        if set_u ~= set_v
            MST_edges(edge_count+1, :) = [CH_XY(u, :), CH_XY(v, :)];
            edge_count = edge_count + 1;
            parent(set_u) = set_v;
        end
    end
end

function p = find_parent(parent, i)
    if parent(i) == i
        p = i;
    else
        p = find_parent(parent, parent(i));
    end
end






