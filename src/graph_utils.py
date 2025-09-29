import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from pagerank_nibble import page_rank_nibble
from collections import Counter, OrderedDict
from itertools import combinations
from collections import defaultdict
import collections
#random.seed(42)

def link_prediction(random_edges, graph, df, testing_file_path, mode):
    or_same_cluster = []
    re_same_cluster = []
    total_jaccard_sim = []
    total_or_mean = []
    total_red_mean = []
    
    for i in range(len(random_edges)):
        red_graph = graph.new_graph_removing_receipts_from_df(df, random_edges[i])
        print("Number of edges:", graph.number_of_edges())
        print("Number of edges:", red_graph.number_of_edges())
        product1 = graph.get_product_to_index(random_edges[i][0])
        product2 = graph.get_product_to_index(random_edges[i][1])
        or_count, red_count, jaccard_similarity, or_mean, red_mean = compare_clusters(graph, red_graph,[product1, product2], mode)
        or_same_cluster.append(or_count)
        re_same_cluster.append(red_count)
        total_jaccard_sim.append(jaccard_similarity)
        total_or_mean.append(or_mean)
        total_red_mean.append(red_mean)
        print(or_count, red_count, jaccard_similarity)
    with open(testing_file_path, 'a') as f:
        f.write(f"or_cluster: {or_same_cluster}\n")
        f.write(f"re_cluster: {re_same_cluster}\n")
        f.write(f"jaccard_sim: {total_jaccard_sim}\n")
        f.write(f"or_edge_weights_mean: {total_or_mean}\n")
        f.write(f"red_edge_weights_mean: {total_red_mean}\n")

def plot_degree_distribution(graph):
    degree_sequence = [d for n, d in graph.degree()]
    degree_count = collections.Counter(degree_sequence)
    degrees, counts = zip(*sorted(degree_count.items())) 

    plt.figure(figsize=(14, 6))
    plt.plot(degrees, counts, marker='o', linestyle='', color='#FF6347', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)  
    #plt.title('Degree Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True)
    plt.xticks(fontsize=12, fontweight='medium')
    plt.yticks(fontsize=12, fontweight='medium')
    plt.tight_layout()
    plt.savefig(f'{graph.t_min}-{graph.t_max}_edge_degree_frequency.png')
    plt.show()

import random

def extract_random_edges(data, val, dev, num, file_path, random_sel="False"):
    filtered_keys = [k for k, v in data.items() if val - dev <= v <= val + dev]
    local_random = random.Random() if random_sel else random.Random(42)
    if len(filtered_keys) < num:
        with open(file_path, 'a') as f:
            f.write(f"trovati solo {len(filtered_keys)} nodi \n")

    return local_random.sample(filtered_keys, min(num, len(filtered_keys)))


def query(client):
    #query = f"SELECT DISTINCT(products) FROM grouped_products WHERE products NOT LIKE '% %';"
    query = f"SELECT cod_prod AS cod_prod, any(descr_liv1) AS descr_liv1, any(descr_liv2) AS descr_liv2, any(descr_liv3) AS descr_liv3, any(descr_liv4) AS descr_liv4, any(descr_rep) AS descr_rep, any(replaceRegexpOne(descr_forn, '^[0-9]+\\s+', '')) AS descr_forn FROM dati_scontrini GROUP BY cod_prod;"
    result = client.query(query)
    print(len(result.result_rows)) 
    return result

# def calulate_edge_weights(client, table_name):
#     query = f"SELECT id_sc, arrayStringConcat(groupArray(cod_prod), ' ') AS products FROM {table_name} GROUP BY id_sc;"
#     result = client.query(query)
#     edge_weights = defaultdict(int)
#     for row in result.result_rows: 
#         products = list(set(row[1].split(' ')))
#         #print(products)
#         for i in range(len(products)):
#             for j in range(i + 1, len(products)):
#                 edge = (products[i], products[j])
#                 edge_weights[edge] += 1
#     return edge_weights

def compute_edges(rows):
    ew = defaultdict(int)
    for row in rows:
        products = list(set(row[1].split(' ')))
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                edge = tuple(sorted((products[i], products[j])))
                ew[edge] += 1
    return ew

def calculate_edge_weights(client, table_name, mode="all", split_ratio=(0.7, 0.10, 0.20), seed=42):
    client.query("SET max_bytes_before_external_group_by=500000000")
    query = f"SELECT id_sc, arrayStringConcat(groupUniqArray(cod_prod), ' ') AS products FROM {table_name} GROUP BY id_sc;"
    result = client.query(query)
    rows = result.result_rows

    if mode == "all":
        return compute_edges(rows)

    elif mode == "split":
        random.seed(seed)
        random.shuffle(rows)

        if isinstance(split_ratio, float):  # only train/test
            split_index = int(len(rows) * split_ratio)
            rows_train, rows_test = rows[:split_index], rows[split_index:]
            return compute_edges(rows_train), compute_edges(rows_test)

        elif isinstance(split_ratio, (tuple, list)) and len(split_ratio) == 3:
            train_ratio, val_ratio, test_ratio = split_ratio
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
                "Train/Val/Test ratios must sum to 1.0"

            n = len(rows)
            idx_train = int(n * train_ratio)
            idx_val = idx_train + int(n * val_ratio)

            rows_train = rows[:idx_train]
            rows_val = rows[idx_train:idx_val]
            rows_test = rows[idx_val:]

            return (
                compute_edges(rows_train),
                compute_edges(rows_val),
                compute_edges(rows_test)
            )

        else:
            raise ValueError("split_ratio must be float (train/test) or tuple of 3 values (train/val/test)")

    else:
        raise ValueError("Mode must be 'all' or 'split'")


# def calculate_edge_weights(client, table_name, mode="all", split_ratio=0.7, seed=42):
#     client.query("SET max_bytes_before_external_group_by=500000000")
#     #query = f"SELECT id_sc, arrayStringConcat(groupArray(cod_prod), ' ') AS products FROM {table_name} GROUP BY id_sc;"
#     query = f"SELECT id_sc, arrayStringConcat(groupUniqArray(cod_prod), ' ') AS products FROM {table_name} GROUP BY id_sc;"
#     result = client.query(query)
#     rows = result.result_rows

#     if mode == "all":
#         edge_weights = compute_edges(rows)
#         return edge_weights

#     elif mode == "split":
#         random.seed(seed)
#         random.shuffle(rows)
#         split_index = int(len(rows) * split_ratio)
#         rows_A, rows_B = rows[:split_index], rows[split_index:]

#         edge_weights_A = compute_edges(rows_A)
#         edge_weights_B = compute_edges(rows_B)

#         return edge_weights_A, edge_weights_B

#     else:
#         raise ValueError("Mode must be 'all' or 'split'")


def display_edge_weight_distribution(edge_weights):
    weights = [weight for _, weight in edge_weights.items()]
    weight_counts = Counter(weights)
    sorted_weight_counts = OrderedDict(sorted(weight_counts.items()))

    weights = list(sorted_weight_counts.keys())
    print(weights[-1])
    counts = list(sorted_weight_counts.values())

    # Create the plot with customized style
    plt.figure(figsize=(14, 6))
    plt.plot(weights, counts, marker='o', linestyle='', color='#FF6347', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2)

    # Log-log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Customize x-axis and y-axis labels
    plt.xlabel('Weights', fontsize=14, labelpad=10)
    plt.ylabel('Frequency', fontsize=14, labelpad=10)
    
    # Title with consistent style
    plt.title('Edge Weight Distribution', fontsize=16, fontweight='bold', pad=20)

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

def print_fit_power_law_on_CCDF(values, counts, empirical_ccdf, title, name_plot):
    expanded_values = np.repeat(values, counts)
    print(f"Length of expanded values: {len(expanded_values)}")
    fit = powerlaw.Fit(expanded_values, discrete=True)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    print(f'Power law exponent: {alpha}')
    print(f'xmin: {xmin}')

    plt.figure(figsize=(14, 6))
    fit.power_law.plot_pdf(color='r', linestyle='--', label=f'Power law fit (xmin={xmin:.2f}, alpha={alpha:.2f})')
   
    #plt.step(values, empirical_ccdf, where='post', marker='o', markerfacecolor='yellow', linestyle='', color='b', label='Empirical data')
    plt.plot(values, counts, marker='o', linestyle='', color='#FF6347', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{name_plot}.png')
    plt.show()

def print_fit_power_law(values, counts, name_plot):
    expanded_values = np.repeat(values, counts)
    print(f"Length of expanded values: {len(expanded_values)}")

    # Fit the power law to the expanded values
    fit = powerlaw.Fit(expanded_values, discrete=True, xmin=3, xmax=2000)
    alpha = fit.power_law.alpha
    xmin = int(fit.power_law.xmin)
    xmax = int(fit.power_law.xmax)

    print(f'Power law exponent: {alpha}')
    print(f'xmin: {xmin}')
    print(f'xmax: {xmax}')

    # Create a figure for plotting
    plt.figure(figsize=(14, 6))

    # Create the first axis (ax1) for frequency (counts)
    ax1 = plt.gca()

    ax1.plot(values, counts, marker='o', linestyle='', color='gold', markersize=8,
             markerfacecolor='#4682B4', linewidth=2, label='Empirical Data')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Degree Values', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    ax2 = ax1.twinx()
    fit.power_law.plot_pdf(color='r', linestyle='--', label = fr'Power law fit ($x_{{\mathrm{{min}}}}$ = {xmin}, $x_{{\mathrm{{max}}}}$ = {xmax}, α = {alpha:.2f})', ax=ax2)
    ax2.set_ylabel('PDF', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    # Title and legends
    #plt.title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True)

    # Show the legend
    ax1.legend(loc='lower left', fontsize=14)
    ax2.legend(loc='upper right', fontsize=14)

    # Save the plot
    plt.savefig(f'{name_plot}.png')

    # Display the plot
    plt.show()

def fit_powerlaw_on_degree_distribution(G):
    degrees = [degree for _, degree in G.degree()]
    degree_counts = Counter(degrees)
    sorted_degree_counts = OrderedDict(sorted(degree_counts.items()))
    degrees = np.array(list(sorted_degree_counts.keys()))
    counts = np.array(list(sorted_degree_counts.values()))
    # Calculate empirical CCDF
    #empirical_ccdf = 1.0 - np.cumsum(counts) / np.sum(counts)
    #print_fit_power_law_on_CCDF(degrees, counts, empirical_ccdf, "Power Law Fit on Degree Distribution", "power_law_fit_degree")
    print_fit_power_law(degrees, counts, "power_law_fit_degree")

def fit_powerlaw_on_edge_distribution(edge_weights):
    weights = [weight for _, weight in edge_weights.items()]
    weight_counts = Counter(weights)
    sorted_weight_counts = OrderedDict(sorted(weight_counts.items()))
    weights = np.array(list(sorted_weight_counts.keys()))
    counts = np.array(list(sorted_weight_counts.values()))
    #empirical_ccdf = 1.0 - np.cumsum(counts) / np.sum(counts)  
    print_fit_power_law(weights, counts, "power_law_fit_edgeweight")

def remove_random_edges(graph, percentage):
    graph_copy = graph.copy()
    num_edges_to_remove = int(percentage * graph.number_of_edges() / 100)
    edges = list(graph.edges())   
    edges_to_remove = random.sample(edges, num_edges_to_remove)
    graph_copy.remove_edges_from(edges_to_remove)    
    return graph_copy

import time
def calculate_clusters(graph, selected_nodes, mode):
    beta = 0.85
    clusters = []
    times = []
    for node in selected_nodes:
        start_time = time.time()
        _, cluster = page_rank_nibble(graph, beta, mode, node)
        end_time = time.time()
        times.append(end_time - start_time)
        print("or len: ", len(cluster))
        clusters.append(cluster)
    return clusters, times

def metric_calculation(graph, original_cluster, reduced_cluster):
    result = []
    sensitivity = []
    precision = []
    #cluster_ratio = []
    #jaccard_sim = []
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
        #print("or", len(or_cluster))
        count = sum(1 for elem in red_cluster if elem in or_cluster)
        sensitivity.append(count / len(or_cluster))
        precision.append(count / len(red_cluster))
        #cluster_ratio.append(len(red_cluster) / len(or_cluster))
        #jaccard_sim.append(compute_jaccard_sim(or_cluster, red_cluster))
        or_mean_degree = graph.mean_degree(or_cluster) 
        red_mean_degree = graph.mean_degree(red_cluster) 
    return result, sensitivity, precision, or_mean_degree, red_mean_degree

def sample_nodes_within_degree_range(graph, degree_min, degree_max, x, random_sel="False"):
    eligible_nodes = [node for node, degree in graph.degree() if degree_min < degree <= degree_max]
    # Step 2: If fewer than x nodes match, adjust x to avoid an error
    if len(eligible_nodes) < x:
        with open('test_epsiloncompute.txt', 'a') as f:
            f.write(f"Warning: Only {len(eligible_nodes)} nodes available within the degree range.")
        x = len(eligible_nodes)
    local_random = random.Random() if random_sel else random.Random(42)
    sampled_nodes = local_random.sample(eligible_nodes, x)
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

def compute_mean_edge_weights(graph, cluster):
    edge_weights = graph.get_subgraph(cluster).size(weight='weight')
    num_edges = graph.get_subgraph(cluster).number_of_edges()
    return edge_weights / num_edges if num_edges > 0 else 0

def compare_clusters(graph, reduced_graph, products, mode):
    or_cluster, _ = calculate_clusters(graph, [products[0]], mode)
    red_cluster, _ = calculate_clusters(reduced_graph, [products[0]], mode)
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
    or_mean = compute_mean_edge_weights(graph, or_cluster[0])
    red_mean = compute_mean_edge_weights(reduced_graph, red_cluster[0])
    return or_count, red_count, jaccard_similarity, or_mean, red_mean

import ast
def validation(graph, reduced_graph, mode, num_nodes=0, random="False"):
    degree_min = [10]
    degree_max = [float('inf')]
    for min_d, max_d in zip(degree_min, degree_max):
        if random == "True":
            selected_nodes = sample_nodes_within_degree_range(graph, min_d, max_d, int(num_nodes /len(degree_min)), random) 
            with open("../data/selected_nodes.txt", "w") as f:
                f.write(str(selected_nodes))
        else:
            with open("../data/selected_nodes.txt") as f:
                selected_nodes = ast.literal_eval(f.read())
            selected_nodes = [graph.product_to_index[node] for node in selected_nodes]
            
        single_node_cluster = []
        original_cluster, times = calculate_clusters(graph, selected_nodes, mode)
        reduced_cluster, _ = calculate_clusters(reduced_graph, selected_nodes, mode) 
        print("len or cl:", len(original_cluster))
        print("len red cl:", len(reduced_cluster))
        result, sensitivity, precision, or_mean_degree, red_mean_degree = metric_calculation(graph, original_cluster, reduced_cluster)
        single_node_cluster = len(original_cluster) - len(result)
        len_cluster = [len(cluster) for cluster in original_cluster]
        #selected_nodes = [graph.index_to_product[node] for node in selected_nodes]
        or_cluster = [cluster[:min(15, len(cluster))] for cluster in original_cluster]
        red_cluster = [cluster[:min(15, len(cluster))] for cluster in reduced_cluster]

        weigh = "w" if mode == "weighted" else "unw"
        with open(f'../Results/{weigh}_test_epsilon_performances.txt', 'a') as f:
            #f.write(f"selected_nodes: {selected_nodes}\n")
            f.write(f"graph: {graph.t_min} - {graph.t_max}\n")
            f.write(f"CCR: {result}\n")
            f.write(f"sensitivity: {sensitivity}\n")
            f.write(f"precision: {precision}\n")
            f.write(f"lenghts: {len_cluster}\n")
            f.write(f"single_node_cluster: {single_node_cluster}\n")
            f.write(f"times: {times}\n")
            #f.write(f"jaccard_similarity: {jaccard_sim}\n")
            f.write(f"or_cluster: {or_cluster}\n")
            f.write(f"red_cluster: {red_cluster}\n")
            f.write(f"or_mean_degree: {or_mean_degree}\n")
            f.write(f"red_mean_degree: {red_mean_degree}\n")
    return
import torch
def create_pyg_data_from_networkx(G, weight_attr='weight'):
    """
    Create PyG Data object from NetworkX graph with proper formatting
    """
    # Get all edges
    edges = list(G.edges())
    
    # Create edge_index in correct format [2, num_edges]
    edge_list = []
    edge_weights = []
    
    for u, v in edges:
        edge_list.append([u, v])  # Assuming nodes are 0-indexed integers
        edge_data = G.get_edge_data(u, v)
        weight = edge_data.get(weight_attr, 1.0) if edge_data else 1.0
        edge_weights.append(weight)
    
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    # Create Data object
    from torch_geometric.data import Data
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=G.number_of_nodes())
    
    return data

def cosine_similarity(u, v):
    return torch.dot(u, v) / (torch.norm(u) * torch.norm(v))

def prune_graph(G, node_embeddings, weight_threshold=None, sim_threshold=0.7):
    edges_to_remove = []

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)

        # if w > weight_threshold:
        #     continue  # always keep
        # else:
        sim = cosine_similarity(node_embeddings[u], node_embeddings[v])
        if sim < sim_threshold:  # too far in embedding space
            edges_to_remove.append((u, v))

    G.remove_edges_from(edges_to_remove)
    return G

def extract_non_adjacent_nodes(G, cluster, x):
    neighbors_x = set(G.neighbors(x))
    non_adjacent = [node for node in cluster if node != x and node not in neighbors_x]
    #top_30 = non_adjacent[:min(30,len(non_adjacent))]
    return non_adjacent