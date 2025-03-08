import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import json
import pickle
import os 
from utils import page_rank_nibble
from collections import Counter, OrderedDict
from itertools import combinations
from collections import defaultdict
#random.seed(42)

def load_graph(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            product_graph = pickle.load(f)
        print("Graph loaded successfully.")
    else:
        print(f"File {file_path} does not exist.")
        return
    return product_graph

def create_graph_subtracting_edgeweights(G, weights_to_remove):
    new_G = nx.Graph() 
    new_G.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        new_weight = data.get('weight', 0) - weights_to_remove[(u,v)]
        if new_weight > 1:
            new_G.add_edge(u, v, weight=new_weight)
    return new_G

def new_graph_removing_receipts(G, dict, client):
    valid_edges = [(dict[u], dict[v], w) for u, v, w in G.edges(data="weight") if 100 <= w <= 200] #Choose a specific pair of products that were sold together
    random_edge = list(random.choice(valid_edges) if valid_edges else None)
    #query=f"SELECT id_sc, arrayStringConcat(groupArray(cod_prod), ' ') AS products FROM dati_scontrini GROUP BY id_sc HAVING match(products, '(^|\\s){random_edge[0]}(\\s|$)') AND match(products, '(^|\\s){random_edge[1]}(\\s|$)');"
    query = f"SELECT * FROM grouped_products WHERE match(products, '(^|\\s){random_edge[0]}(\\s|$)') AND match(products, '(^|\\s){random_edge[1]}(\\s|$)');"
    result = client.query(query)
    weights_to_remove = defaultdict(int)
    for row in result.result_rows:
        products = list(set(row[1].split(' ')))
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                edge = (products[i], products[j])
                weights_to_remove[edge] += 1
    new_graph = create_graph_subtracting_edgeweights(G, weights_to_remove)
    return new_graph, random_edge[0], random_edge[1]


def calulate_edge_weights(client, table_name):
    # query_avg_products = f"WITH product_counts AS (SELECT id_sc, length(groupArray(cod_prod)) AS num_products FROM {table_name} GROUP BY id_sc ) SELECT AVG(num_products) AS avg_products_per_id_sc FROM  product_counts;"
    
    # result = client.query(query_avg_products)
    # print("mean:", result.result_rows)
    # query= F" SELECT DISTINCT descr_rep FROM {table_name};"
    # result = client.query(query)
    # print("result:", result.result_rows)
    query = f"SELECT id_sc, arrayStringConcat(groupArray(cod_prod), ' ') AS products FROM {table_name} GROUP BY id_sc;"
    result = client.query(query)

    edge_weights = defaultdict(int)
    for row in result.result_rows:
        products = list(set(row[1].split(' ')))
        #print(products)
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                edge = (products[i], products[j])
                edge_weights[edge] += 1
    return edge_weights

def create_graph(edge_weights, t_min, t_max):

    product_graph_A = nx.Graph()
    product_to_index = {}
    index_to_product = {}
    current_index = 0

    for edge, weight in edge_weights.items():
        if weight >= t_min: #and weight < t_max:
            product_i, product_j = edge

            for p in [product_i, product_j]:
                if p not in product_to_index:
                    product_to_index[p] = current_index
                    index_to_product[current_index] = p
                    current_index += 1

            product_i_index = product_to_index[product_i]
            product_j_index = product_to_index[product_j]

            product_graph_A.add_edge(product_i_index, product_j_index, weight=weight)   
    return product_graph_A, product_to_index, index_to_product

# def display_edge_weight_distribution(edge_weights):
#     weights = [ weight for _, weight in edge_weights.items()]
#     weight_counts = Counter(weights)
#     sorted_weight_counts = OrderedDict(sorted(weight_counts.items()))

#     weights = list(sorted_weight_counts.keys())
#     counts = list(sorted_weight_counts.values())

#     plt.figure(figsize=(10, 6))
#     plt.plot(weights, counts, marker='o', linestyle='-', color='b')
#     plt.xscale('log')
#     plt.yscale('log')

#     plt.xlabel('Weights')
#     plt.ylabel('Frequency')
#     plt.title('Frequency of each Edge Weight (Log-Log Scale)')
#     plt.grid(True)

#     plt.savefig('edge_weights_frequency.png')
#     plt.show()

def display_edge_weight_distribution(edge_weights):
    weights = [weight for _, weight in edge_weights.items()]
    weight_counts = Counter(weights)
    sorted_weight_counts = OrderedDict(sorted(weight_counts.items()))

    weights = list(sorted_weight_counts.keys())
    print(weights[-1])
    counts = list(sorted_weight_counts.values())

    # Create the plot with customized style
    plt.figure(figsize=(10, 6))
    plt.plot(weights, counts, marker='o', linestyle='--', color='#FF6347', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2)

    # Log-log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Customize x-axis and y-axis labels
    plt.xlabel('Weights', fontsize=14, labelpad=10)
    plt.ylabel('Frequency', fontsize=14, labelpad=10)
    
    # Title with consistent style
    plt.title('Frequency of Each Edge Weight (Log-Log Scale)', fontsize=16, fontweight='bold', pad=20)

    # Grid customization (matching the second function)
    plt.grid(True)

    # Customize x and y ticks
    plt.xticks(fontsize=12, fontweight='medium')
    plt.yticks(fontsize=12, fontweight='medium')

    # Set a light gray background for consistency
    plt.gca().set_facecolor('#f5f5f5')

    # Tight layout for better spacing
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('edge_weights_frequency_custom.png')
    plt.show()

def fit_powerlaw_on_edge_distribution_CDDF(edge_weights):
    # Get weights and counts
    weights = [weight for _, weight in edge_weights.items()]
    weight_counts = Counter(weights)
    sorted_weight_counts = OrderedDict(sorted(weight_counts.items()))

    weights = np.array(list(sorted_weight_counts.keys()))
    counts = np.array(list(sorted_weight_counts.values()))

    # Calculate empirical CCDF
    empirical_ccdf = 1.0 - np.cumsum(counts) / np.sum(counts)

    # Expand weights based on their counts (for correct fitting)
    expanded_weights = np.repeat(weights, counts)
    print(len(expanded_weights))
    # Fit the power law on the expanded weights
    fit = powerlaw.Fit(expanded_weights, discrete=True)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    print(f'Power law exponent: {alpha}')
    print(f'xmin: {xmin}')

    # Plot the CCDF and the power law fit
    plt.figure(figsize=(14, 6))

    # Power law fit CCDF (plot only for x >= xmin)
    fit.power_law.plot_ccdf(color='r', linestyle='--', label=f'Power law fit (xmin={xmin:.2f}, alpha={alpha:.2f})')

    # Plot empirical CCDF
    plt.step(weights, empirical_ccdf, where='post', marker='o', markerfacecolor='yellow',linestyle='-', color='b', label='Empirical data')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Weights')
    plt.ylabel('CCDF')
    plt.title('Power Law Fit on Edge Weight Distribution (Log-Log Scale)', fontsize=16, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True)
    plt.savefig('power_law_fit.png')
    plt.show()
    
def calculate_graph_info(edge_weights, t_min):
    graph_info = {}
    graph_info[int(t_min)] = {}
    for offset in [51, 41, 31, 21, 11]:
        
        product_graph_A, product_to_index, index_to_product = create_graph(edge_weights, int(t_min), offset)

        print("Number of nodes:", product_graph_A.number_of_nodes())
        print("Number of edges:", product_graph_A.number_of_edges())

        graph_info[int(t_min)][offset] = {
            "number_of_nodes": product_graph_A.number_of_nodes(),
            "number_of_edges": product_graph_A.number_of_edges()
        }
    print(graph_info)
    with open("graph_info.json", 'a') as f:
        json.dump(graph_info, f, indent=4)

def test_different_epsilon(args, table_name, client, graph, index_to_product, product_to_index):
    clusters_info = {}
    clusters_info[args.seed] = {}
    query= f"SELECT descr_prod FROM {table_name} WHERE cod_prod IN ({args.seed}) LIMIT 1;"
    result = client.query(query)
    split_result = result.result_rows[0][0].split()
    des = ' '.join(split_result[1:])
    clusters_info[args.seed]["description"] = des
    clusters_info[args.seed]["t_min"] = args.t_min
    clusters_info[args.seed]["t_max"] = args.t_max
    for i, epsilon in enumerate([0.01, 0.001, 0.0001, 0.00002,0.00001]):
    #for i, epsilon in enumerate([0.1, 0.03, 0.015, 0.01, 0.009]):
        n = graph.number_of_nodes()
        phi = 0.1
        beta = 0.9
        #epsilon = 0.00001 #diminuire 
        mode = args.mode
        seed, cluster = page_rank_nibble(graph, n, phi, beta, epsilon, mode, product_to_index[args.seed])
        
        ids = [index_to_product[index] for index in cluster]
        ids = ', '.join(map(str, ids))
        print("len: ", len(ids.split()))
        print(ids)
        query= f"SELECT DISTINCT descr_prod FROM {table_name} WHERE cod_prod IN ({ids});"
        result = client.query(query)
        descriptions = []
        for row in result.result_rows:
            row_split = row[0].split()
            des = ' '.join(row_split[1:])
            descriptions.append(des)
        clusters_info[args.seed][epsilon] = (len(ids.split()), descriptions)
        print(i)

    with open("clusters_info.json", 'a') as f:
        json.dump(clusters_info, f, indent=4)

def remove_random_edges(graph, percentage):
    graph_copy = graph.copy()
    num_edges_to_remove = int(percentage * graph.number_of_edges() / 100)
    edges = list(graph.edges())   
    edges_to_remove = random.sample(edges, num_edges_to_remove)
    graph_copy.remove_edges_from(edges_to_remove)    
    return graph_copy

def calculate_clusters(graph, reduced_graph, selected_nodes, mode, c):
    #n = graph.number_of_nodes()
    beta = 0.9
    c = 0.00001
    #epsilon = 2e-05
    original_cluster = []
    reduced_cluster = []
    for node in selected_nodes:
        #print(node)
        seed, cluster = page_rank_nibble(graph, beta, c, mode, node)
        original_cluster.append(cluster)
        seed, cluster = page_rank_nibble(reduced_graph, beta, c, mode, node)
        reduced_cluster.append(cluster)
    return original_cluster, reduced_cluster

def metric_calculation(graph, original_cluster, reduced_cluster):
    result = []
    sensitivity = []
    precision = []
    #cluster_ratio = []
    jaccard_sim = []
    for or_cluster, red_cluster in zip(original_cluster, reduced_cluster):
        den = num = 0
        for u, v in combinations(or_cluster, 2):
            if graph.has_edge(u, v):
                den+= 1
                elements_present = np.isin([u, v], red_cluster)
                if elements_present.all():
                    num+= 1
        if den != 0:
            result.append(num / den)
        print("or", len(or_cluster))
        count = sum(1 for elem in red_cluster if elem in or_cluster)
        sensitivity.append(count / len(or_cluster))
        precision.append(count / len(red_cluster))
        #cluster_ratio.append(len(red_cluster) / len(or_cluster))
        jaccard_sim.append(compute_jaccard_sim(or_cluster, red_cluster))
    return result, sensitivity, precision, jaccard_sim

def sample_nodes_within_degree_range(graph, degree_min, degree_max, x, mode):
    if mode == "weighted":
        eligible_nodes = [node for node, degree in graph.degree(weight='weight') if degree_min < degree <= degree_max]
    else:
        eligible_nodes = [node for node, degree in graph.degree() if degree_min < degree <= degree_max]
    # Step 2: If fewer than x nodes match, adjust x to avoid an error
    if len(eligible_nodes) < x:
        with open('w_testing.txt', 'a') as f:
            f.write(f"Warning: Only {len(eligible_nodes)} nodes available within the degree range.")
        x = len(eligible_nodes)
    random.seed(42)
    sampled_nodes = random.sample(eligible_nodes, x)
    return sampled_nodes

def are_present(cluster, products):
    print("cluster:", cluster)
    print("products:", products)
    if products[0] in cluster and products[1] in cluster:
        return True
    return False
    
def compute_jaccard_sim(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2  
    union = set1 | set2
    return len(intersection) / len(union) if len(union) > 0 else 0

def compare_clusters(graph, reduced_graph, products, mode, epsilon):
    or_cluster, red_cluster = calculate_clusters(graph, reduced_graph, [products[0]], mode, epsilon)
    print("lengths:", len(or_cluster[0]), len(red_cluster[0]))
    or_count = 0 
    red_count = 0
    jaccard_similarity = []
    for o_c, r_c in zip(or_cluster, red_cluster):
        if are_present(o_c, products):
            or_count+=1
        if are_present(r_c, products):
            red_count+=1
        jaccard_similarity.append(compute_jaccard_sim(o_c,r_c))
    return or_count, red_count, jaccard_similarity

#def test_compute_epsilon(graph, reduced_graph, num_nodes, mode):


def validation(graph, reduced_graph, num_nodes, mode):
    degree_min = [0,10, 100, 1000]
    degree_max = [10, 100, 1000, 10000]
    c = 0.00001
    with open('test_epsiloncompute.txt', 'a') as f:
        f.write(f"constant: {c}\n")
    for min, max in zip(degree_min, degree_max):
        selected_nodes = sample_nodes_within_degree_range(graph, min, max, int(num_nodes /len(degree_min)), mode) 
        #epsilon_values = [8e-05, 7e-05, 6e-05, 5e-05]
        #print(selected_nodes)
        #print("range: ", min, " - ", max)
        with open('test_epsiloncompute.txt', 'a') as f:
            f.write(f"range: {min} - {max}\n")
            #f.write(f"epsilon_values: {epsilon_values}\n")
        single_node_cluster = []
        #for epsilon in epsilon_values:
        original_cluster, reduced_cluster = calculate_clusters(graph, reduced_graph, selected_nodes, mode,c) 
        result, sensitivity, precision, jaccard_sim = metric_calculation(graph, original_cluster, reduced_cluster)
        single_node_cluster = len(original_cluster) - len(result)
            #dev.append(np.std(result, ddof=0) if (len(result)<=1) else np.std(result, ddof=1))
            #accuracy_dev.append(np.std(accuracy, ddof=0) if (len(accuracy)<=1) else np.std(accuracy, ddof=1))
            #cluster_ratio_dev.append(np.std(cluster_ratio, ddof=0) if (len(cluster_ratio)<=1) else np.std(cluster_ratio, ddof=1))
            #final_results.append(np.mean(result))
            #final_accuracy.append(np.mean(accuracy))
            #final_cluster_ratio.append(np.mean(cluster_ratio))
        len_cluster = [len(cluster) for cluster in original_cluster]
            #cluster_len_mean.append(np.mean(len_cluster))
            #cluster_len_dev.append(np.std(len_cluster))
        with open('test_epsiloncompute.txt', 'a') as f:
            f.write(f"CCR: {result}\n")
            f.write(f"sensitivity: {sensitivity}\n")
            f.write(f"precision: {precision}\n")
            f.write(f"lenghts: {len_cluster}\n")
            #f.write(f"CLR: {cluster_ratio}\n")
            f.write(f"single_node_cluster: {single_node_cluster}\n")
            f.write(f"jaccard_similarity: {jaccard_sim}\n")
        # #with open('unweighted_metrics.txt', 'a') as f:
        # with open('test_unw_metrics.txt', 'a') as f:
        #     f.write(f"std_devs: {dev}\n")
        #     f.write(f"metric_values: {final_results}\n")
        #     f.write(f"epsilon_values: {epsilon_values}\n")
        #     f.write(f"accuracy: {final_accuracy}\n")
        #     f.write(f"accuracy_dev: {accuracy_dev}\n")
        #     f.write(f"cluster_ratio: {final_cluster_ratio}\n")
        #     f.write(f"cluster_ratio_dev: {cluster_ratio_dev}\n")
        #     f.write(f"single_node_cluster: {single_node_cluster}\n")
        #     f.write(f"cluster_len_mean: {cluster_len_mean}\n")
        #     f.write(f"cluster_len_dev: {cluster_len_dev}\n")
    return

def testing_epsilon(graph, reduced_graph, num_nodes, mode):
    selected_nodes = random.sample(list(graph.nodes()), num_nodes)
    epsilon_values = [0.0001, 8e-05, 5e-05, 2e-05]
    with open('w_testing.txt', 'a') as f:
        f.write(f"epsilon_values: {epsilon_values}\n")
    for epsilon in epsilon_values:
        print(epsilon)
        original_cluster, reduced_cluster = calculate_clusters(graph, reduced_graph, selected_nodes, mode, epsilon) 
        result, sensitivity, cluster_ratio = metric_calculation(graph, original_cluster, reduced_cluster)
        single_node_cluster = len(original_cluster) - len(result)
        len_clusters = [len(cluster) for cluster in original_cluster]

        with open('w_testing.txt', 'a') as f:
            f.write(f"CCR: {result}\n")
            f.write(f"sensitivity: {sensitivity}\n")
            f.write(f"lenghts: {len_clusters}\n")
            f.write(f"CLR: {cluster_ratio}\n")
            f.write(f"single_node_cluster: {single_node_cluster}\n")
    return

def metrics_plot_with_std_devs(unweighted_metric_values, weighted_metric_values, epsilon_values, 
                               unweighted_std_devs, weighted_std_devs):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot for unweighted graph with error bars (standard deviations as errors)
    plt.errorbar(
        epsilon_values, unweighted_metric_values, yerr=unweighted_std_devs, 
        fmt='-o', capsize=5, capthick=2, elinewidth=2, 
        color='#FF6347', ecolor='#FF4500', markerfacecolor='#4682B4',
        markersize=8, linestyle='--', linewidth=2, label="Unweighted Metric"
    )
    
    # Plot for weighted graph with error bars (standard deviations as errors)
    plt.errorbar(
        epsilon_values, weighted_metric_values, yerr=weighted_std_devs, 
        fmt='-s', capsize=5, capthick=2, elinewidth=2, 
        color='#4682B4', ecolor='#1E90FF', markerfacecolor='#FF6347',
        markersize=8, linestyle='-', linewidth=2, label="Weighted Metric"
    )

    # Use a logarithmic scale for the x-axis
    plt.xscale('log')

    # Set x-ticks only at the epsilon values corresponding to the metric values
    plt.xticks(epsilon_values, fontsize=12, fontweight='medium')

    # Customize the axes and background
    plt.xlabel('Epsilon Values (Log Scale)', fontsize=14, labelpad=10)
    plt.ylabel('Metric Values', fontsize=14, labelpad=10)
    plt.title('Metrics vs Epsilon with Standard Deviation', fontsize=16, fontweight='bold', pad=20)

    # Add grid (log scale)
    plt.grid(True)

    # Customize ticks for y-axis
    plt.yticks(fontsize=12, fontweight='medium')

    # Adding a light grey background
    plt.gca().set_facecolor('#f5f5f5')

    # Adding legend to differentiate unweighted and weighted metrics
    plt.legend(fontsize=12, loc='best', fancybox=True, shadow=True)

    # Adding tight layout for better spacing
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('metrics_plot.png')

    # Show the plot
    plt.show()


# def metrics_plot_with_std_devs(metric_values, epsilon_values, std_devs):
#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     # Plot with error bars (standard deviations as errors)
#     plt.errorbar(
#         epsilon_values, metric_values, yerr=std_devs, 
#         fmt='-o', capsize=5, capthick=2, elinewidth=2, 
#         color='#FF6347', ecolor='#4682B4', markerfacecolor='#4682B4',
#         markersize=8, linestyle='--', linewidth=2, label="Metric with Std Dev"
#     )

#     # Use a logarithmic scale for the x-axis
#     plt.xscale('log')

#     # Set x-ticks only at the epsilon values corresponding to the metric values
#     plt.xticks(epsilon_values, fontsize=12, fontweight='medium')

#     # Customize the axes and background
#     plt.xlabel('Epsilon Values (Log Scale)', fontsize=14, labelpad=10, fontweight='bold')
#     plt.ylabel('Metric Values', fontsize=14, labelpad=10, fontweight='bold')
#     plt.title('Metric vs Epsilon (Log Scale) with Standard Deviation', fontsize=16, fontweight='bold', pad=20)

#     # Add grid (log scale)
#     plt.grid(True, which="both", linestyle=':', linewidth=0.8)

#     # Customize ticks for y-axis
#     plt.yticks(fontsize=12, fontweight='medium')

#     # Adding a light grey background
#     plt.gca().set_facecolor('#f5f5f5')

#     # Adding legend
#     plt.legend(fontsize=12, loc='best', fancybox=True, shadow=True)

#     # Adding tight layout for better spacing
#     plt.tight_layout()

    
#     plt.savefig('metrics_plot.png')
#     # Show the plot
#     plt.show()