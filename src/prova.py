import os
import argparse
import pickle
from collections import defaultdict
from dotenv import load_dotenv
#import clickhouse_connect
from graph import Graph
from graph_utils import create_pyg_data_from_networkx, prune_graph, calculate_clusters, extract_non_adjacent_nodes, calculate_edge_weights, display_edge_weight_distribution
from LightGCN import *
import torch
import ast
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def compute_set_metrics(ground_truth, predictions):
    precisions, recalls, f1s = [], [], []
    for node in ground_truth:
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
    

def get_similar_products(graph, node_embeddings, product_idx, threshold=0.7):
    product_embedding = node_embeddings[product_idx]
    similarities = torch.nn.functional.cosine_similarity(product_embedding.unsqueeze(0), node_embeddings, dim=1)
    
    # Get indices of nodes connected to product_idx (including itself)
    connected = set(graph.get_neighbors(product_idx))
    connected.add(product_idx)  # Exclude self as well

    # Find indices with similarity > threshold and not connected
    similar_indices = [idx for idx, sim in enumerate(similarities) if sim > threshold and idx not in connected]
    similar_scores = similarities[similar_indices]

    return similar_indices, similar_scores

def get_edge_weights_from_nx(G):
    return {(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}

def converting_metadata(metadata):
    df = pd.DataFrame.from_dict(metadata, orient="index")
    print(df.head())
    encoders = {}
    for level in ["descr_liv1", "descr_liv2", "descr_liv3", "descr_liv4"]:
        le = LabelEncoder()
        df[level] = le.fit_transform(df[level])
        encoders[level] = le
    print(df.head())
    print(df.columns)
    with open("../data/metadata_labels.pkl", "wb") as f:
        pickle.dump(df, f)

def simple_align_labels(node_labels_df, product_to_node_map):
    """
    Simple alignment - removes nodes that don't exist in the map
    """
    print('node labels len')
    rows, columns = node_labels_df.shape
    print(f"Rows: {rows}, Columns: {columns}")
    print('product to node map len', len(product_to_node_map.keys()))
    # Get only the product IDs that exist in both the DataFrame and the map
    common_products = set(node_labels_df.index) & set(product_to_node_map.keys())
    
    if not common_products:
        raise ValueError("No common products found between node_labels_df and product_to_node_map")
    
    # Filter DataFrame and map node IDs
    aligned_df = node_labels_df.loc[list(common_products)].copy()
    aligned_df.index = [product_to_node_map[product_id] for product_id in aligned_df.index]
    
    # Sort by node ID
    aligned_df = aligned_df.sort_index()
    print('node labels len')
    rows, columns = aligned_df.shape
    print(f"Rows: {rows}, Columns: {columns}")
    return aligned_df

def main():

    parser = argparse.ArgumentParser(description='Insert thresholds for the graph cluster algorithm')
    parser.add_argument('--load', default= "True", help='Specify if load the graph or create a new one')
    parser.add_argument('--t_min', default=0, help='Specify the min threshold')
    parser.add_argument('--t_max', default=float('inf'), help='Specify the max threshold')
    parser.add_argument('--mode', default="weighted", choices=["weighted","unweighted"], help='Specify whether to use weighted or unweighted graph')
    parser.add_argument('--seed', help='Specify id of the seed node')
    parser.add_argument('--task', default="performances", choices=["predictions", "performances","frequent_items", "prova"], help='Specify the task to perform')
    parser.add_argument('--edge_weight', default=50, type=int, help='Specify the edge weight for link prediction task')
    parser.add_argument('--random', default="False", help='Specify if use random selection of nodes/edges for the selected task')
    args = parser.parse_args()
    load_dotenv()

    # client = clickhouse_connect.get_client(
    #         host=os.getenv('CLICKHOUSE_HOST'),
    #         port=int(os.getenv('CLICKHOUSE_PORT')),
    #         username=os.getenv('CLICKHOUSE_USER'),
    #         password=os.getenv('CLICKHOUSE_PASSWORD'),
    #         database=os.getenv('CLICKHOUSE_DATABASE')
    #         )

    if args.task == "prova":
        with open("../data/metadata_labels.pkl", "rb") as f:
            metadata = pickle.load(f)
        print("Metadata loaded")
        table_name = 'dati_scontrini'
        if (os.path.exists(f"../data/train_edge_weights.pkl") and os.path.exists(f"../data/test_edge_weights.pkl")):
            with open(f"../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)
            with open(f"../data/val_edge_weights.pkl", "rb") as f:
                val_edge_weights = pickle.load(f)
            with open(f"../data/test_edge_weights.pkl", "rb") as f:
                test_edge_weights = pickle.load(f)
        else:
            train_edge_weights, test_edge_weights = calculate_edge_weights(client, table_name, "split")
            with open(f"../data/train_edge_weights.pkl", "wb") as f:
                pickle.dump(train_edge_weights, f)
            with open(f"../data/test_edge_weights.pkl", "wb") as f:
                pickle.dump(test_edge_weights, f)
            
        
        train_graph = Graph()
        val_graph = Graph()
        test_graph = Graph()

        print("arrivata qui")
        train_graph.create_graph(train_edge_weights, int(args.t_min), float(args.t_max))
        val_graph.create_graph(val_edge_weights, int(args.t_min), float(args.t_max))
        test_graph.create_graph(test_edge_weights, int(args.t_min), float(args.t_max))
        print("Train Graph - Number of nodes:", train_graph.number_of_nodes())
        print("Train Graph - Number of edges:", train_graph.number_of_edges())
        print("Val Graph - Number of nodes:", val_graph.number_of_nodes())
        print("Val Graph - Number of edges:", val_graph.number_of_edges())
        print("Test Graph - Number of nodes:", test_graph.number_of_nodes())
        print("Test Graph - Number of edges:", test_graph.number_of_edges())
        
        with open("../data/selected_nodes.txt") as f:
            selected_nodes = ast.literal_eval(f.read())
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
        #print(train_neighbors)
        #print(test_neighbors)
        print(ground_truth)
        print("ottenute etichette")
        # metadata = simple_align_labels(metadata, train_graph.product_to_index)
        # node_embeddings = train_lightgcn(train_graph, metadata, val_graph)
        # print("Embeddings shape:", node_embeddings.shape)  # [num_nodes, 64]

        data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)

        # Load the saved weights
        model.load_state_dict(torch.load("../LightGCN/lightgcn_model.pth"))
        model.eval()
        node_embeddings = model.get_embeddings(data.edge_index, edge_weight=data.edge_attr)
        print("Embeddings shape:", node_embeddings.shape)  # [num_nodes, 64]

        G_pruned = prune_graph(train_graph, node_embeddings, weight_threshold=20, sim_threshold=0.7)
        print("G pruned - Number of nodes:", G_pruned.number_of_nodes())
        print("G pruned - Number of edges:", G_pruned.number_of_edges())
        
        #G_pruned_edge_weights = get_edge_weights_from_nx(G_pruned.graph)
        #display_edge_weight_distribution(G_pruned_edge_weights)

        selected_nodes_indices = [G_pruned.get_product_to_index(node) for node in selected_nodes]
        embeddings_predictions = defaultdict(set)
        for i,node_idx in enumerate(selected_nodes_indices):
            similar_indices, similar_scores = get_similar_products(train_graph,node_embeddings, node_idx)
            embeddings_predictions[selected_nodes[i]] = set([G_pruned.get_index_to_product(x) for x in similar_indices])
        #similar_indices = similar_indices.tolist()
        #print(f"Products similar to product 0: {[train_graph.get_index_to_product(x) for x in similar_indices]}")
        # intersections = {key: ground_truth[key] & embeddings_predictions.get(key, set()) for key in set(ground_truth) | set(embeddings_predictions)}
        # print("Intersections:", intersections)
        # non_empty_count = sum(1 for intersection_set in intersections.values() if intersection_set)
        # print(f"Number of nodes with non-empty intersection: {non_empty_count} out of {len(selected_nodes)}")

        clusters, _ = calculate_clusters(G_pruned, selected_nodes_indices, args.mode)
        clusters_filtered = []
        #tensor_data = torch.tensor(clusters, dtype=torch.float32)
        #torch.save(tensor_data, "../data/pagerank_features.pt")
        #exit(0)
        #top_15_clusters = [[x for x in clusters[i][1:16]] for i in range(len(clusters))]
        clusters_predictions = defaultdict(set)
        clusters_predictions_filtered = defaultdict(set)
        for i, node in enumerate(selected_nodes):
            non_adjacent = extract_non_adjacent_nodes(train_graph, clusters[i], G_pruned.get_product_to_index(selected_nodes[i]))
            print("len non adiacenti:", len(non_adjacent))
            if len(non_adjacent) > 0:
               clusters[i] = non_adjacent
               clusters_filtered.append(filter_clusters(G_pruned, non_adjacent, selected_nodes[i], node_embeddings))
            clusters_predictions[node] = set([G_pruned.get_index_to_product(x) for x in clusters[i]])
            clusters_predictions_filtered[node] = set([G_pruned.get_index_to_product(x) for x in clusters_filtered[i]])
            #print("ground truth", ground_truth[node])
            #print("top 15", top_15_clusters[node])
        # intersections_clusters = {key: ground_truth[key] & clusters_predictions.get(key, set()) for key in set(ground_truth) | set(clusters_predictions)}
        # intersections_clusters_filtered = {key: ground_truth[key] & clusters_predictions_filtered.get(key, set()) for key in set(ground_truth) | set(clusters_predictions_filtered)}
        # non_empty_count_clusters = sum(1 for intersection_set in intersections_clusters.values() if intersection_set)
        # non_empty_count_clusters_filtered = sum(1 for intersection_set in intersections_clusters_filtered.values() if intersection_set)
        # print(f"Number of nodes with non-empty intersection using clusters: {non_empty_count_clusters} out of {len(selected_nodes)}")
        # print(f"Number of nodes with non-empty intersection using filtered clusters: {non_empty_count_clusters_filtered} out of {len(selected_nodes)}")
        for name, preds in [
            ("Embeddings", embeddings_predictions),
            ("Clusters", clusters_predictions),
            ("Clusters Filtered", clusters_predictions_filtered)
        ]:
            precision, recall, f1 = compute_set_metrics(ground_truth, preds)
            print(f"{name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
if __name__ == "__main__":
    main()