% Load the saved data from wsn_results.mat
load('wsn_results.mat');

% Ensure optimal_k does not exceed the number of elements in best_solutions
optimal_k = min(optimal_k, numel(best_solutions));
sink_node = [250, 250]; % Sink node coordinates

% Initialize variables for total recovery time
total_recovery_time = 0;

% Initialize an empty list to store the new cluster heads
new_cluster_heads_list = cell(optimal_k, 1);
% Initialize cluster_data_list to hold the data for each cluster
cluster_data_list = cell(optimal_k, 1);

% Debug: Display the size of best_solutions
disp(['Size of best_solutions: ', num2str(numel(best_solutions))]);

% Populate cluster_data_list with the data for each cluster
for clusterIdx = 1:optimal_k
    % Debug: Check if the index is within bounds
    if clusterIdx > numel(best_solutions)
        error('Index exceeds the number of array elements in best_solutions. Index must not exceed %d.', numel(best_solutions));
    end
    
    % Debug: Display the current cluster index and corresponding best_solution
    disp(['Processing cluster index: ', num2str(clusterIdx)]);
    disp(['Best solution for this cluster: ', mat2str(best_solutions{clusterIdx})]);

    cluster_data_list{clusterIdx} = struct('nodes', nodes(idx == clusterIdx, :), ...
                                           'CH', cell2mat(best_solutions(clusterIdx)));
end

% Loop through each cluster and apply the GA
for clusterIdx = 1:optimal_k
    % Extract cluster data
    cluster_data = cluster_data_list{clusterIdx};
    clusterNodes = cluster_data.nodes;
    clusterCH = cluster_data.CH; % Fixed: changed from cluster_data.best_solutions

    % Initialize downed CH
    downed_CH = clusterCH;

    % Maximum number of generations
    max_generations = 600;

    % Initialize population
    population_size = 5;
    population = randi([1, size(clusterNodes, 1)], population_size, size(clusterNodes, 1));

    % Fitness function
    fitness_function = @(pop) calculate_fitness(pop, clusterNodes, clusterCH);

    % Initialize variables to store the best solution and its fitness
    best_solution = [];
    best_fitness = -Inf;

    % Initialize fitness history
    fitness_history = zeros(max_generations, 1);

    % Visualization
    figure;
    xlabel('Generation');
    ylabel('Fitness Value');
    title('Fitness Value vs. Generation');

    for generation = 1:max_generations
        % Evaluate fitness of each individual in the population
        fitness_values = fitness_function(population);

        % Find the best solution in the current population
        [current_best_fitness, index] = max(fitness_values);
        current_best_solution = population(index, :);

        % Update the best solution and its fitness
        if current_best_fitness > best_fitness
            best_fitness = current_best_fitness;
            best_solution = current_best_solution;
        end

        % Update fitness history
        fitness_history(generation) = best_fitness;

        % Visualization
        plot(1:generation, fitness_history(1:generation), 'b');
        xlabel('Generation');
        ylabel('Fitness Value');
        title('Fitness Value vs. Generation');
        hold on;
        pause(0.005);

        % Select parents for crossover (tournament selection)
        num_parents = 2;
        parents = zeros(num_parents, size(clusterNodes, 1));
        for i = 1:num_parents
            tournament = randi(population_size, 1, 2);
            [~, idx] = max(fitness_values(tournament));
            parents(i, :) = population(tournament(idx), :);
        end

        % Perform crossover (single-point crossover)
        crossover_point = randi([1, size(clusterNodes, 1) - 1]);
        offspring = zeros(population_size, size(clusterNodes, 1));
        for i = 1:2:population_size
            parent1 = parents(mod(i, num_parents) + 1, :);
            parent2 = parents(mod(i + 1, num_parents) + 1, :);
            offspring(i, :) = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
            offspring(i + 1, :) = [parent2(1:crossover_point), parent1(crossover_point+1:end)];
        end

        % Perform mutation (randomly select a node and change it to a random node)
        mutation_rate = 0.1;
        for i = 1:population_size
            if rand < mutation_rate
                mutation_point = randi(size(clusterNodes, 1));
                offspring(i, mutation_point) = randi(size(clusterNodes, 1));
            end
        end

        % Update population with offspring
        population = offspring;
    end

    % Calculate recovery time for the current cluster
    recovery_time = max_generations * 0.005;  % Assuming each generation takes 5 milliseconds
    total_recovery_time = total_recovery_time + recovery_time;

    % Display the best solution and its fitness
    fprintf('Cluster %d - Best solution: Node %d, Best fitness value: %.4f\n', clusterIdx, best_solution(1), best_fitness);

    % Store the new cluster head for this cluster
    new_cluster_heads_list{clusterIdx} = clusterNodes(best_solution(1), :);
end

% Display the total recovery time for all clusters
fprintf('Total recovery time for all clusters: %.4f seconds\n', total_recovery_time);

% Now new_cluster_heads_list contains the new cluster heads for each cluster
% You can access the new cluster head for each cluster by indexing into the list

% Visualize the entire network and MST with marked downed CH and new CH
figure;
hold on;

% Define colors for each cluster
colors = lines(optimal_k);

% Plot each cluster with different colors
for clusterIdx = 1:optimal_k
    clusterNodes = cluster_data_list{clusterIdx}.nodes;
    scatter(clusterNodes(:,1), clusterNodes(:,2), 50, colors(clusterIdx,:), 'filled');
end

% Mark downed CHs with red 'x'
for clusterIdx = 1:optimal_k
    downed_CH = cluster_data_list{clusterIdx}.CH;
    scatter(downed_CH(1), downed_CH(2), 100, 'rx', 'LineWidth', 2);
end

% Mark new CHs with green 'o'
for clusterIdx = 1:optimal_k
    new_CH = new_cluster_heads_list{clusterIdx};
    scatter(new_CH(1), new_CH(2), 100, 'go', 'LineWidth', 2);
end

% Add MST edges
sink_node = [0, 0]; % Assuming the sink node is at (0,0), change as needed
MST_edges = kruskal_mst(new_cluster_heads_list, sink_node);
for i = 1:size(MST_edges, 1)
    plot([MST_edges(i,1), MST_edges(i,3)], [MST_edges(i,2), MST_edges(i,4)], 'k-', 'LineWidth', 1.5);
end
% Plot connection between cluster nodes and cluster head
    for j = 1:size(cluster_nodes, 1)
        plot([cluster_nodes(j, 1), selected_ch(1)], [cluster_nodes(j, 2), selected_ch(2)], 'k');
    end
xlabel('X Coordinate');
ylabel('Y Coordinate');
title('Network Visualization with Downed and New Cluster Heads');
legend('Clusters', 'Downed CH', 'New CH', 'MST');
grid on;
hold off;

% Fitness function
function fitness = calculate_fitness(node, node_coordinates, clusterCH)
    population_size = size(node, 1);
    num_nodes = size(node, 2);
    fitness = zeros(population_size, 1);
    for j = 1:population_size
        for i = 1:num_nodes
            if node(j, i) == find(ismember(node_coordinates, clusterCH, 'rows'))
                % Penalize the solution if it selects the current CH
                fitness(j) = fitness(j) - 1000;
            else
                distance_to_CH = norm(node_coordinates(node(j, i), :) - clusterCH);
                distance_to_others = sum(sqrt(sum((node_coordinates - node_coordinates(node(j, i), :)).^2, 2)));
                fitness(j) = fitness(j) + 1 / (1 + distance_to_CH + distance_to_others);
            end
        end
    end
end

% Kruskal's MST function
function MST_edges = kruskal_mst(best_solutions, sink_node)
    % Convert cell array to matrix for easier handling
    sink_node = [250, 250]; % Sink node coordinates
    CH_XY = [];
    for i = 1:length(best_solutions)
        if iscell(best_solutions{i})
            CH_XY = [CH_XY; cell2mat(best_solutions{i})];
        else
            CH_XY = [CH_XY; best_solutions{i}];
        end
    end
    CH_XY = [CH_XY; sink_node]; % Add sink node to the list of CHs

    % Number of CHs including the sink node
    numCH = size(CH_XY, 1);

    % Create edge list with distances
    edgeList = [];
    for i = 1:numCH
        for j = i+1:numCH
            distance = norm(CH_XY(i, :) - CH_XY(j, :));
            edgeList = [edgeList; i, j, distance];
        end
    end

    % Sort edge list by distance
    edgeList = sortrows(edgeList, 3);

    % Initialize MST edges and disjoint set
    MST_edges = [];
    parent = 1:numCH;
    rank = zeros(1, numCH);

    % Find function for disjoint set
    function root = find(x)
        if parent(x) ~= x
            parent(x) = find(parent(x));
        end
        root = parent(x);
    end

    % Union function for disjoint set
    function union(x, y)
        rootX = find(x);
        rootY = find(y);
        if rootX ~= rootY 
                if rank(rootX) > rank(rootY)
                parent(rootY) = rootX;
            elseif rank(rootX) < rank(rootY)
                parent(rootX) = rootY;
            else
                parent(rootY) = rootX;
                rank(rootX) = rank(rootX) + 1;
            end
        end
    end

    % Kruskal's algorithm to construct MST
    numEdges = 0;
    i = 1;
    while numEdges < numCH - 1
        node1 = edgeList(i, 1);
        node2 = edgeList(i, 2);
        if find(node1) ~= find(node2)
            MST_edges = [MST_edges; CH_XY(node1, :), CH_XY(node2, :)];
            union(node1, node2);
            numEdges = numEdges + 1;
        end
        i = i + 1;
    end
end
