import pandas as pd
import pickle
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit
import clickhouse_connect
from utils import page_rank_nibble
from collections import Counter, OrderedDict
import numpy as np
import json
import powerlaw
import os
import random
from itertools import combinations

def calulate_edge_weights(client, table_name):

    query = f"SELECT id_sc, arrayStringConcat(groupArray(cod_prod), ' ') AS products FROM {table_name} GROUP BY id_sc LIMIT 10000;"
    result = client.query(query)

    edge_weights = defaultdict(int)

    for row in result.result_rows:
        products = row[1].split(' ')
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
        if weight > t_min and weight < t_max:
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


def display_edge_weight_distribution(edge_weights):
    weights = [ weight for _, weight in edge_weights.items()]
    weight_counts = Counter(weights)
    sorted_weight_counts = OrderedDict(sorted(weight_counts.items()))

    weights = list(sorted_weight_counts.keys())
    counts = list(sorted_weight_counts.values())

    # # Create a line plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(weights, counts, marker='o', linestyle='-', color='b')

    # # Set log-log scale
    # plt.xscale('log')
    # plt.yscale('log')

    # # Adding labels and title
    # plt.xlabel('Weights')
    # plt.ylabel('Frequency')
    # plt.title('Frequency of each Edge Weight (Log-Log Scale)')
    # plt.grid(True)

    # # Save the plot
    # plt.savefig('edge_weights_frequency.png')
    # plt.show()
    
    fit = powerlaw.Fit(weights, discrete=True)
    
    # Extract the exponent of the power law
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    # Display the results
    print(f'Power law exponent: {alpha}')
    print(f'xmin: {xmin}')

    # Plot the fit on top of the empirical data
    plt.figure(figsize=(10, 6))
    fit.power_law.plot_ccdf(color='r', linestyle='--', label='Power law fit')
    plt.plot(weights, counts, marker='o', linestyle='-', color='b', label='Empirical data')
    
    # Set log-log scale
    plt.xscale('log')
    plt.yscale('log')

    # Adding labels and title
    plt.xlabel('Weights')
    plt.ylabel('Frequency')
    plt.title('Power Law Fit on Edge Weight Distribution (Log-Log Scale)')
    plt.legend()
    plt.grid(True)

    # Save the plot
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

def calculate_clusters(graph, reduced_graph, selected_nodes):
    n = graph.number_of_nodes()
    phi = 0.1
    beta = 0.9
    epsilon = 2e-05
    original_cluster = []
    reduced_cluster = []
    for node in selected_nodes:
        seed, cluster = page_rank_nibble(graph, n, phi, beta, epsilon, "unweighted", node)
        original_cluster.append(cluster)
        seed, cluster = page_rank_nibble(reduced_graph, n, phi, beta, epsilon, "unweighted", node)
        reduced_cluster.append(cluster)
    return original_cluster, reduced_cluster

def validation(graph, reduced_graph, num_nodes):
    selected_nodes = random.sample(list(graph.nodes()), num_nodes)
    print(selected_nodes)
    original_cluster, reduced_cluster = calculate_clusters(graph, reduced_graph, selected_nodes)
    result = 0
    for or_cluster, red_cluster in zip(original_cluster, reduced_cluster):
        den = num = 0
        for u, v in combinations(or_cluster, 2):
            if graph.has_edge(u, v):
                den+= 1
                elements_present = np.isin([u, v], red_cluster)
                if elements_present.all():
                    num+= 1
        result += num / den 
    return result / num_nodes 

def main():

    parser = argparse.ArgumentParser(description='Insert thresholds of the algorithm')
    parser.add_argument('--load', default= "True", help='Specify if load the graph or create a new one')
    parser.add_argument('--t_min', help='Specify the min threshold')
    parser.add_argument('--t_max', help='Specify the max threshold')
    parser.add_argument('--mode', choices=["weighted","unweighted"], help='Specify if you want to consider the graph weighted or unweighted')
    parser.add_argument('--seed',help='Specify id seed node')
    args = parser.parse_args()

    if args.load == "False":

        #edge_weights = calulate_edge_weights(client, table_name)
        #with open('edge_weights.pkl', 'wb') as file:
            #pickle.dump(edge_weights, file)
    
        with open('edge_weights.pkl', 'rb') as file:
            edge_weights = pickle.load(file)
        
        product_graph, product_to_index, index_to_product = create_graph(edge_weights, int(args.t_min), int(args.t_max))

        with open("graph.pkl", "wb") as f:
            pickle.dump(product_graph, f)
        
        with open('products_dict.pkl', 'wb') as f:
            pickle.dump({'id_to_index': product_to_index, 'index_to_id': index_to_product}, f)
    
    else:
        file_path = "graph.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                product_graph = pickle.load(f)
            print("Graph loaded successfully.")
        else:
            print(f"File {file_path} does not exist.")
    
    #display_edge_weight_distribution(edge_weights)

    print("Number of nodes:", product_graph.number_of_nodes())
    print("Number of edges:", product_graph.number_of_edges())
    percent = 30
    reduced_graph = remove_random_edges(product_graph, percent)

    print("Number of nodes:", reduced_graph.number_of_nodes())
    print("Number of edges:", reduced_graph.number_of_edges())
    value = validation(product_graph, reduced_graph, 1)
    print(value)
    exit(0)

    host = 'localhost'
    port = 8123
    database = 'mydb'
    user = 'evision'
    psw= 'Evision!'
    table_name = 'dati_scontrini'
    
    client = clickhouse_connect.get_client(host=host, port=port, username=user, password=psw, database=database)
     
    #test_different_epsilon(args, table_name, client, product_graph, index_to_product, product_to_index)

    # print("seed node:", seed)
    # #print("Identified cluster:", cluster)
    # ids = [index_to_product[index] for index in cluster]
    # print(len(ids))
    # ids = ', '.join(map(str, ids))
    # query= f"SELECT DISTINCT descr_prod FROM {table_name} WHERE cod_prod IN ({ids});"
    # result = client.query(query)
    # print("cluster products: ",result.result_rows)
    # query= f"SELECT DISTINCT descr_prod FROM {table_name} WHERE cod_prod IN ({index_to_product[seed]});"
    # result = client.query(query)
    # print("seed product:", result.result_rows)
    client.close()

if __name__ == "__main__":
    main()