import cProfile
import os
import argparse
import pickle
from collections import defaultdict
from tracemalloc import start
from dotenv import load_dotenv
#import clickhouse_connect
from graph import Graph
from graph_utils import *
from plots_utils import plot_non_adjacent_proportion
from LightGCN import *
import torch
import ast
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from CategoryDistibutionComparison import CategoryDistributionComparator
from LightGCNPruningTuner import *
from PageRankNibbleTuner import *


def extract_nodes_for_testing(G_pruned, num_nodes):
    """
    Extracts nodes from G_pruned according to degree ranges.
    Splits evenly into 4 buckets of size num_nodes//4.

    Ranges:
        0-10, 10-100, 100-1000, 1000+
    Ensures the final list is ordered in ascending degree.
    """
    per_range = num_nodes // 4

    ranges = [
        (1, 10),
        (10, 100),
        (100, 1000),
        (1000, float("inf"))
    ]

    selected_nodes = []
    degrees = dict(G_pruned.degree())

    for low, high in ranges:
        # filter nodes in range
        bucket = [
            node for node, deg in degrees.items()
            if low <= deg < high
        ]
        # sort bucket by degree
        bucket_sorted = sorted(bucket, key=lambda n: degrees[n])
        # take first per_range
        selected_nodes.extend(bucket_sorted[:per_range])

    # sort final list by degree again, to ensure overall order
    selected_nodes = sorted(selected_nodes, key=lambda n: degrees[n])

    return selected_nodes

# i want to visualize the portion of connected vs non connected predictions for both methods - and find out why some products have more non connected predictions than others (like how often they are pursached - in how many receipts - or how many connections they have in the original graph)

def calculate_labels(train_graph, test_graph, selected_nodes):
    train_neighbors = defaultdict(set)
    test_neighbors = defaultdict(set)
    ground_truth = defaultdict(set)
    selected_nodes = selected_nodes[:20]
    for node in selected_nodes:
        t_n =train_graph.get_neighbors(train_graph.get_product_to_index(node))
        train_neighbors[node] = set([train_graph.get_index_to_product(n) for n in t_n])
        t_n = test_graph.get_neighbors(test_graph.get_product_to_index(node))
        test_neighbors[node] = set([test_graph.get_index_to_product(n) for n in t_n])
        ground_truth[node] = set(test_neighbors[node] - train_neighbors[node])
    return ground_truth

def compute_set_metrics(ground_truth, predictions, keys):
    precisions, recalls, f1s = [], [], []
    for node in keys:
        gt = ground_truth[node]
        pred = predictions.get(node, set())
        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    return avg_precision, avg_recall, avg_f1

def filter_clusters(G_pruned, non_adjacent, selected_node, node_embeddings):
    ref_idx = G_pruned.get_product_to_index(selected_node)
    ref_emb = node_embeddings[ref_idx]
    filtered = []
    for idx in non_adjacent:
        sim = torch.nn.functional.cosine_similarity(ref_emb, node_embeddings[idx], dim=0)
        if sim > 0.7:
            filtered.append(idx)
    return filtered 
    

def get_similar_products(node_embeddings, product_idx, mode, threshold=0.8, graph=None):
    product_embedding = node_embeddings[product_idx]
    similarities = torch.nn.functional.cosine_similarity(product_embedding.unsqueeze(0), node_embeddings, dim=1)

    if mode == "all":
        # Return all indices with similarity > threshold (including connected)
        similar_indices = [idx for idx, sim in enumerate(similarities) if sim > threshold]
    elif mode == "non_adjacent":
        # Exclude connected nodes
        connected = set(graph.get_neighbors(product_idx))
        connected.add(product_idx)
        similar_indices = [idx for idx, sim in enumerate(similarities) if sim > threshold and idx not in connected]
    else:
        raise ValueError("Unknown mode: choose 'all' or 'non_adjacent'")

    similar_scores = similarities[similar_indices]
    return similar_indices, similar_scores

def get_edge_weights_from_nx(G):
    return {(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}

def converting_metadata(metadata):
    df = pd.DataFrame.from_dict(metadata, orient="index")
    print(df.head())
    encoders = {}
    for level in ["descr_liv1", "descr_liv2", "descr_liv3", "descr_liv4", "descr_rep", "descr_forn"]:
        le = LabelEncoder()
        df[level] = le.fit_transform(df[level])
        encoders[level] = le
    print(df.head())
    print(df.columns)
    with open("../data/metadata_labels.pkl", "wb") as f:
        pickle.dump(df, f)


def main():

    parser = argparse.ArgumentParser(description='Insert thresholds for the graph cluster algorithm')
    parser.add_argument('--load', default= "True", help='Specify if load the graph or create a new one')
    parser.add_argument('--mode', default="weighted", choices=["weighted","unweighted"], help='Specify whether to use weighted or unweighted graph')
    parser.add_argument('--task', default="comparison", choices=["training", "pruning","pruning_tuning", "comparison", "prn_tuning", "lightgcn_tuning", "link_prediction", "performances"], help='Specify the task to perform')
    parser.add_argument('--random', default="False", help='Specify if use random selection of nodes/edges for the selected task')
    parser.add_argument('--edge_weight', default=10, type=int, help='Specify the edge weight for link prediction task')
    args = parser.parse_args()
    load_dotenv() 
        
    if not os.path.exists("../LightGCN/lightgcn_model_049.pth") or args.task == "training":
        if (os.path.exists("../data/train_edge_weights.pkl") and os.path.exists("../data/val_edge_weights.pkl")):
            with open("../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)
            with open("../data/val_edge_weights.pkl", "rb") as f:
                val_edge_weights = pickle.load(f)
        else:
            # client = clickhouse_connect.get_client(
            #         host=os.getenv('CLICKHOUSE_HOST'),
            #         port=int(os.getenv('CLICKHOUSE_PORT')),
            #         username=os.getenv('CLICKHOUSE_USER'),
            #         password=os.getenv('CLICKHOUSE_PASSWORD'),
            #         database=os.getenv('CLICKHOUSE_DATABASE')
            #         )
            table_name = 'dati_scontrini'
            train_edge_weights, test_edge_weights, val_edge_weights = calculate_edge_weights(client, table_name, "split")
            with open("../data/train_edge_weights.pkl", "wb") as f:
                pickle.dump(train_edge_weights, f)
            with open("../data/test_edge_weights.pkl", "wb") as f:
                pickle.dump(test_edge_weights, f)
            with open("../data/val_edge_weights.pkl", "wb") as f:
                pickle.dump(val_edge_weights, f)

        with open ("../data/metadata_labels.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        #metadata = simple_align_labels(metadata, train_graph.product_to_index)


        if not (os.path.exists("../data/train_graph.pkl")):
            train_graph = Graph()
            train_graph.create_graph(train_edge_weights, node_labels=metadata) 
        else:
            with open("../data/train_graph.pkl", "rb") as f:
                train_graph = pickle.load(f)

        val_graph = Graph()
        val_graph.create_graph(val_edge_weights)
        
        node_embeddings = train_lightgcn(train_graph, val_graph)
        print("Embeddings shape:", node_embeddings.shape)
    
    if not (os.path.exists("../data/train_pruned.pkl")) or args.task == "pruning":
        print("Pruning the graph...")
        if not (os.path.exists("../data/train_graph.pkl")):
            with open("../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)

            with open ("../data/metadata_labels.pkl", "rb") as f:
                metadata = pickle.load(f)

            train_graph = Graph()
            train_graph.create_graph(train_edge_weights, node_labels=metadata) 
            with open("../data/train_graph.pkl", "wb") as f:
                pickle.dump(train_graph, f)
        else:
            with open("../data/train_graph.pkl", "rb") as f:
                train_graph = pickle.load(f)
        data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)

        # Load the saved weights
        model.load_state_dict(torch.load("../LightGCN/lightgcn_model_049.pth"))
        model.eval()
        node_embeddings = model.get_embedding(data.edge_index, edge_weight=data.edge_attr)
        print("Embeddings shape:", node_embeddings.shape)  # [num_nodes, 64]

        G_pruned = prune_graph(train_graph, node_embeddings, sim_threshold=0.5)
        print("G pruned - Number of nodes:", G_pruned.number_of_nodes())
        print("G pruned - Number of edges:", G_pruned.number_of_edges()) 
        with open("../data/train_pruned.pkl", "wb") as f:
            pickle.dump(G_pruned, f) 

    if args.task == "pruning_tuning":
        if not (os.path.exists("../data/train_graph.pkl")):
            with open ("../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)
            with open ("../data/metadata_labels.pkl", "rb") as f:
                metadata = pickle.load(f)
            
            train_graph = Graph()
            train_graph.create_graph(train_edge_weights, node_labels=metadata)
        else:
            with open("../data/train_graph.pkl", "rb") as f:
                train_graph = pickle.load(f)

        # data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        # model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)
        # # Load the saved weights
        # model.load_state_dict(torch.load("../LightGCN/lightgcn_model_099.pth"))

        # LightGCN_tuner = LightGCNTuner(model, train_graph, data)
        
        with open(f"../data/train_pruned.pkl", "rb") as f:
            train_pruned = pickle.load(f)
        print("G pruned - Number of nodes:", train_pruned.number_of_nodes())
        print("G pruned - Number of edges:", train_pruned.number_of_edges())
        train_results = compare_node_categories(train_graph)
        pruned_results = compare_node_categories(train_pruned)
        #plot_category_similarity_boxplots(train_results, "TrainGraph")
        plot_category_similarity_boxplots(pruned_results, "Now Pruned Graph")


    elif args.task == "prn_tuning": 
        with open("../data/train_pruned.pkl", "rb") as f:
            G_pruned = pickle.load(f)
        if (os.path.exists(f"../data/test_nodes.pkl")):
            with open(f"../data/test_nodes.pkl", "rb") as f:
                selected_nodes = pickle.load(f)
        else:
            selected_nodes = extract_nodes_for_testing(G_pruned, 200)
            with open(f"../data/test_nodes.pkl", "wb") as f:
                pickle.dump(selected_nodes, f)
        print("len selected nodes:", len(selected_nodes))
        pagerank_tuner = PageRankNibbleTuner(G_pruned, mode=args.mode, selected_nodes=selected_nodes)
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        portion_non_connected, times = pagerank_tuner.tune_parameters()
        profiler.disable()
        profiler.print_stats(sort='cumtime')
        #profiler.dump_stats('profile_results.prof')
        plot_non_adjacent_proportion(portion_non_connected, "prn")

        plt.figure(figsize=(7, 5))
        plt.ylabel("Time (s)")
        plt.plot(times, color='red', marker='o', label=args.mode, linestyle='--')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='best', fancybox=True, shadow=True)
        plt.savefig(f"time_comparison.png")
        plt.show()

    elif args.task == "link_prediction":
        #Link Prediction task
        testing_file_path = "../Results/Link prediction/prova.txt"
        weight = args.edge_weight
        dev = 10
        num = 50
        with open("../data/train_pruned.pkl", "rb") as f:
            graph = pickle.load(f)
        print("Number of nodes:", graph.number_of_nodes())
        print("Number of edges:", graph.number_of_edges())
        # with open('../data/train_edge_weights.pkl', 'rb') as file:
        #     edge_weights = pickle.load(file)
        # random_edges = extract_random_edges(edge_weights, weight, dev, num, testing_file_path, args.random)
        edges_with_y_weight = [(graph.get_index_to_product(u), graph.get_index_to_product(v)) for u, v, d in graph.edges(data=True) if d.get('weight', 1) == weight]
        selected_edges = edges_with_y_weight[:num]  # Get the first num edges

        with open(testing_file_path, 'a') as f:
            f.write(f"EDGE_WEIGHT: {weight - dev} - {weight + dev}\n")
        df = pd.read_csv("../data/grouped_products.csv")
        beta = 0.85
        c = 0.002 if args.mode == "unweighted" else 0.00006
        link_prediction(selected_edges, graph, df, testing_file_path, args.mode, beta, c)

    elif args.task == "performances":
        # Performance evaluation with Reduced Graphs
        with open("../data/train_pruned.pkl", "rb") as f:
            graph = pickle.load(f)
        percent = 50
        reduced_graph = remove_random_edges(graph, percent)
        print("Number of nodes in original graph:", graph.number_of_nodes())
        print("Number of edges in original graph:", graph.number_of_edges())
        print("Number of nodes:", reduced_graph.number_of_nodes())
        print("Number of edges:", reduced_graph.number_of_edges())    
        validation(graph, reduced_graph, args.mode, 0.85, 0.002 if args.mode == "unweighted" else 0.00006)

    elif args.task == "lightgcn_tuning":
        with open("../data/train_pruned.pkl", "rb") as f:
            G_pruned = pickle.load(f)
        if (os.path.exists(f"../data/test_nodes.pkl")):
            with open(f"../data/test_nodes.pkl", "rb") as f:
                selected_nodes = pickle.load(f)
        else:
            selected_nodes = extract_nodes_for_testing(G_pruned, 150)
            with open(f"../data/test_nodes.pkl", "wb") as f:
                pickle.dump(selected_nodes, f)
        print("len selected nodes:", len(selected_nodes))

        with open ("../data/train_graph.pkl", "rb") as f:
            train_graph = pickle.load(f)
        data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)
        model.load_state_dict(torch.load("../LightGCN/lightgcn_model_099.pth"))
        model.eval()
        node_embeddings = model.get_embedding(data.edge_index, edge_weight=data.edge_attr)

        portion_non_connected = []
        for node in selected_nodes:
            similar_indices, similar_scores = get_similar_products(node_embeddings, node, "all", threshold=0.6)
            non_adjacent = extract_non_adjacent_nodes(G_pruned, similar_indices, node)
            portion_non_connected.append(len(non_adjacent) / len(similar_indices) if similar_indices else 0)
        
        plot_non_adjacent_proportion(portion_non_connected, "lightgcn")

    elif args.task == "comparison":
        prediction_mode = "non_adjacent"  # or "all"
        with open("../data/test_nodes.pkl", "rb") as f:
            selected_nodes = pickle.load(f)
        #selected_nodes = selected_nodes[:100]
        with open("../data/train_pruned.pkl", "rb") as f:
            G_pruned = pickle.load(f)
        print("Train Graph Pruned - Number of nodes:", G_pruned.number_of_nodes())
        print("Train Graph Pruned - Number of edges:", G_pruned.number_of_edges())
        #G_pruned_edge_weights = get_edge_weights_from_nx(G_pruned.graph)
        #fit_powerlaw_on_edge_distribution(G_pruned_edge_weights)
        #fit_powerlaw_on_degree_distribution(G_pruned)
        #exit(0)
        if not os.path.exists(f"../data/train_graph.pkl"):
            with open(f"../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)
            with open(f"../data/metadata_labels.pkl", "rb") as f:
                metadata = pickle.load(f)
            train_graph = Graph()
            train_graph.create_graph(train_edge_weights, node_labels=metadata)
        else:
            with open(f"../data/train_graph.pkl", "rb") as f:
                train_graph = pickle.load(f)
        if (os.path.exists(f"../data/ground_truth.pkl")):
            with open("../data/ground_truth.pkl", "rb") as f:
                ground_truth = pickle.load(f)
        else:
            if (os.path.exists(f"../data/train_edge_weights.pkl") and os.path.exists(f"../data/test_edge_weights.pkl")):
                with open(f"../data/train_edge_weights.pkl", "rb") as f:
                    train_edge_weights = pickle.load(f)
                with open(f"../data/test_edge_weights.pkl", "rb") as f:
                    test_edge_weights = pickle.load(f)

            
            test_graph = Graph()
            test_graph.create_graph(test_edge_weights)
        
            train_neighbors = defaultdict(set)
            test_neighbors = defaultdict(set)
            ground_truth = defaultdict(set)
            nodes_id = [train_graph.get_index_to_product(n) for n in selected_nodes if n in train_graph.index_to_product]
            for node in nodes_id:
                t_n =train_graph.get_neighbors(train_graph.get_product_to_index(node))
                train_neighbors[node] = set([train_graph.get_index_to_product(n) for n in t_n])

                node_idx = test_graph.get_product_to_index(node)
                if node_idx is None:
                    test_neighbors[node] = set()
                else:
                    t_n = test_graph.get_neighbors(node_idx)
                    filtered_neighbors = [
                        test_graph.get_index_to_product(n)
                        for n in t_n
                        #if test_graph.get_weight(node_idx, n) > 1
                    ]
                    test_neighbors[node] = set(filtered_neighbors)
                candidates = list(set(test_neighbors[node] - train_neighbors[node]))
                gt = [c for c in candidates if c in train_graph.product_to_index]
                ground_truth[node] = set(gt)
            
            with open("../data/ground_truth.pkl", "wb") as f:
                pickle.dump(ground_truth, f)
            print("ottenute etichette e salvate")

        data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)

        # Load the saved weights
        model.load_state_dict(torch.load("../LightGCN/lightgcn_model_049.pth"))
        model.eval()
        node_embeddings = model.get_embedding(data.edge_index, edge_weight=data.edge_attr)
        print("Embeddings shape:", node_embeddings.shape) 

        #selected_nodes_indices = [G_pruned.get_product_to_index(node) for node in selected_nodes]
        embeddings_predictions = defaultdict(set)
        for i, node_idx in enumerate(selected_nodes):
            similar_indices, similar_scores = get_similar_products(node_embeddings, node_idx, prediction_mode, threshold=0.6, graph=train_graph)
            embeddings_predictions[G_pruned.get_index_to_product(node_idx)] = set([G_pruned.get_index_to_product(x) for x in similar_indices])

        if (os.path.exists(f"../data/clusters.pkl")):
            with open("../data/clusters.pkl", "rb") as f:
                clusters = pickle.load(f)
        else: 
            clusters, _ = calculate_clusters(G_pruned, selected_nodes, args.mode, 0.80 , c=0.002 if args.mode == "unweighted" else 0.00006)
            with open("../data/clusters.pkl", "wb") as f:
                pickle.dump(clusters, f)
            print("Clusters calculated and saved")

        clusters_predictions = defaultdict(set)
        for i, node in enumerate(selected_nodes):
            if prediction_mode == "non_adjacent":
                non_adjacent = extract_non_adjacent_nodes(train_graph, clusters[i], node)
                #print("len non adiacenti:", len(non_adjacent))
                #if len(non_adjacent) > 0:
                #    clusters[i] = non_adjacent 
                clusters[i] = non_adjacent

            clusters_predictions[G_pruned.get_index_to_product(node)] = set([G_pruned.get_index_to_product(x) for x in clusters[i]])
        emb_intersections = []
        clu_intersections = []
        for key in selected_nodes:
            key = G_pruned.get_index_to_product(key)
            gt = ground_truth.get(key, set())
            emb_pred = embeddings_predictions.get(key, set())
            clu_pred = clusters_predictions.get(key, set())
            intersection = gt & emb_pred
            emb_intersections.append(len(intersection))
            intersection = gt & clu_pred
            clu_intersections.append(len(intersection))
            #denom = min(len(pred), len(gt))
            #ratio = len(intersection) / denom if denom > 0 else 0.0

        union_predictions = defaultdict(set)
        union_intersections = []

        for key in selected_nodes:
            key = G_pruned.get_index_to_product(key)
            emb_pred = embeddings_predictions.get(key, set())
            clu_pred = clusters_predictions.get(key, set())
            union_pred = emb_pred | clu_pred  # Union of both sets
            union_predictions[key] = union_pred

            gt = ground_truth.get(key, set())
            intersection = gt & union_pred
            union_intersections.append(len(intersection))

        len_predictions_per_range = len(selected_nodes) // 4
        nodes = [G_pruned.get_index_to_product(n) for n in selected_nodes]
        ranges = [
            (0, len_predictions_per_range),
            (len_predictions_per_range, 2 * len_predictions_per_range),
            (2 * len_predictions_per_range, 3 * len_predictions_per_range),
            (3 * len_predictions_per_range, len(selected_nodes))

        ]
        range_labels = ["1-10", "10-100", "100-1000", "1000+"]
        def get_keys(start, end):
            return [nodes[i] for i in range(start, end)]
        
        dist_comp = CategoryDistributionComparator(G_pruned)
        for idx, (start, end) in enumerate(ranges):
            keys = get_keys(start, end)
            divergences = dist_comp.evaluate_methods(pagerank={k: clusters_predictions[k] for k in keys if k in clusters_predictions}, lightgcn={k: embeddings_predictions[k] for k in keys if k in embeddings_predictions}, union={k: union_predictions[k] for k in keys if k in union_predictions})
            dist_comp.plot_results(divergences, prediction_mode, range=range_labels[idx])

        # non_connected_emb = defaultdict(set)
        # non_connected_clu = defaultdict(set)
        # connected_weights_emb = defaultdict(list)
        # connected_weights_clu = defaultdict(list)

        # for key_idx in selected_nodes:
        #     key = train_graph.get_index_to_product(key_idx)
        #     connected = set(train_graph.get_neighbors(key_idx))
            
        #     # Embeddings predictions
        #     emb_pred_indices = [train_graph.get_product_to_index(x) for x in embeddings_predictions[key] if x in train_graph.product_to_index]
        #     non_connected_emb[key] = set([x for x in embeddings_predictions[key] if train_graph.get_product_to_index(x) not in connected])
        #     connected_weights_emb[key] = [
        #         train_graph.get_weight(key_idx, idx)
        #         for idx in emb_pred_indices if idx in connected
        #     ]
            
        #     # Clusters predictions
        #     clu_pred_indices = [train_graph.get_product_to_index(x) for x in clusters_predictions[key] if x in train_graph.product_to_index]
        #     non_connected_clu[key] = set([x for x in clusters_predictions[key] if train_graph.get_product_to_index(x) not in connected])
        #     connected_weights_clu[key] = [
        #         train_graph.get_weight(key_idx, idx)
        #         for idx in clu_pred_indices if idx in connected
        #     ]
        # all_weights_emb = [w for weights in connected_weights_emb.values() for w in weights]
        # all_weights_clu = [w for weights in connected_weights_clu.values() for w in weights]

        # plt.figure(figsize=(10,5))

        # def plot_ccdf(data, label, color):
        #     sorted_data = np.sort(data)
        #     ccdf = 1 - np.arange(len(sorted_data)) / len(sorted_data)
        #     plt.loglog(sorted_data, ccdf, label=label, alpha=0.7, linewidth=2)

        # plot_ccdf(all_weights_emb, 'Embeddings Connected Weights', 'blue')
        # plot_ccdf(all_weights_clu, 'Clusters Connected Weights', 'red')
        # plt.xlabel('Edge Weight')
        # plt.ylabel('P(X ≥ x)')
        # plt.legend()
        # plt.title('Complementary CDF of Edge Weights')
        # plt.grid(True, alpha=0.3)
        # plt.show()

        # # Prepare data for plotting
        # x = np.arange(len(selected_nodes))
        # emb_non_connected = [len(non_connected_emb[node]) for node in selected_nodes]
        # clu_non_connected = [len(non_connected_clu[node]) for node in selected_nodes]
        # emb_connected = [len(embeddings_predictions[node]) - len(non_connected_emb[node]) for node in selected_nodes]
        # clu_connected = [len(clusters_predictions[node]) - len(non_connected_clu[node]) for node in selected_nodes]

        # fig, ax = plt.subplots(figsize=(14, 8))

        # # Plot all 4 lines
        # ax.plot(x, emb_non_connected, label='Embeddings Non-Connected', color='skyblue', linewidth=2, marker='o', markersize=3)
        # ax.plot(x, emb_connected, label='Embeddings Connected', color='dodgerblue', linewidth=2, marker='s', markersize=3)
        # ax.plot(x, clu_non_connected, label='Clusters Non-Connected', color='lightcoral', linewidth=2, marker='^', markersize=3)
        # ax.plot(x, clu_connected, label='Clusters Connected', color='firebrick', linewidth=2, marker='d', markersize=3)

        # ax.set_xlabel('Selected Node Index')
        # ax.set_ylabel('Count')
        # ax.set_title('Connected and Non-Connected Items per Selected Node (Line Plot)')
        # ax.set_xticks(x)
        # ax.set_xticklabels([str(node) for node in selected_nodes], rotation=45)
        # ax.legend()
        # ax.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()


        # from statistics import mean
        # non_connected_emb_lens = [len(v) for _, v in non_connected_emb.items()]
        # non_connected_clu_lens = [len(v) for _, v in non_connected_clu.items()]
        # Print some stats
        # print("Embeddings - Non-connected items mean:", {mean(non_connected_emb_lens)})
        # print("Clusters - Non-connected items mean:", {mean(non_connected_clu_lens)})
        # print("Embeddings - Connected weights stats: min =", min(all_weights_emb), "max =", max(all_weights_emb), "mean =", sum(all_weights_emb)/len(all_weights_emb) if all_weights_emb else 0)
        # print("Clusters - Connected weights stats: min =", min(all_weights_clu), "max =", max(all_weights_clu), "mean =", sum(all_weights_clu)/len(all_weights_clu) if all_weights_clu else 0)
        # exit(0)


        # Aggregate and print metrics for each range
        for idx, (start, end) in enumerate(ranges):
            keys = get_keys(start, end)
            # Intersections
            emb_avg = np.mean([emb_intersections[i] for i in range(start, end)])
            clu_avg = np.mean([clu_intersections[i] for i in range(start, end)])
            union_avg = np.mean([union_intersections[i] for i in range(start, end)])
            print(f"Average intersection len for embeddings_predictions - range {range_labels[idx]}: {emb_avg:.3f}")
            print(f"Average intersection len for clusters_predictions - range {range_labels[idx]}: {clu_avg:.3f}")
            print(f"Average intersection len for union of embeddings and clusters predictions - range {range_labels[idx]}: {union_avg:.3f}")

            print(f"Range - {range_labels[idx]}")
            for name, preds in [
                ("Embeddings", embeddings_predictions),
                ("Clusters", clusters_predictions),
                ("Union", union_predictions)
            ]:
                precision, recall, f1 = compute_set_metrics(ground_truth, preds, keys=keys)
                print(f"{name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        for node in selected_nodes:
            node = G_pruned.get_index_to_product(node)
            print(len(ground_truth[node]), len(embeddings_predictions[node]), len(clusters_predictions[node]), len(union_predictions[node]))

if __name__ == "__main__":
    main()