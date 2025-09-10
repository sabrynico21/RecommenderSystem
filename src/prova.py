import os
import argparse
import pickle
from collections import defaultdict
from dotenv import load_dotenv
import clickhouse_connect
from graph import Graph
from graph_utils import calculate_edge_weights
from LightGCN import LightGCN
from graph_utils import create_pyg_data_from_networkx, prune_graph, calculate_clusters, extract_non_adjacent_nodes
import torch
import ast

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

    client = clickhouse_connect.get_client(
            host=os.getenv('CLICKHOUSE_HOST'),
            port=int(os.getenv('CLICKHOUSE_PORT')),
            username=os.getenv('CLICKHOUSE_USER'),
            password=os.getenv('CLICKHOUSE_PASSWORD'),
            database=os.getenv('CLICKHOUSE_DATABASE')
            )

    if args.task == "prova":
        table_name = 'dati_scontrini'
        if (os.path.exists(f"../data/train_edge_weights.pkl") and os.path.exists(f"../data/test_edge_weights.pkl")):
            with open(f"../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)
            with open(f"../data/test_edge_weights.pkl", "rb") as f:
                test_edge_weights = pickle.load(f)
        else:
            train_edge_weights, test_edge_weights = calculate_edge_weights(client, table_name, "split")
            with open(f"../data/train_edge_weights.pkl", "wb") as f:
                pickle.dump(train_edge_weights, f)
            with open(f"../data/test_edge_weights.pkl", "wb") as f:
                pickle.dump(test_edge_weights, f)
        
        train_graph = Graph()
        test_graph = Graph()
        print("arrivata qui")
        train_graph.create_graph(train_edge_weights, int(args.t_min), float(args.t_max))
        test_graph.create_graph(test_edge_weights, int(args.t_min), float(args.t_max))
        print("Train Graph - Number of nodes:", train_graph.number_of_nodes())
        print("Train Graph - Number of edges:", train_graph.number_of_edges())
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

        data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)

        # Load the saved weights
        model.load_state_dict(torch.load("../LightGCN/lightgcn_model.pth"))
        model.eval()
        node_embeddings = model.get_embeddings(data.edge_index, edge_weight=data.edge_attr)
        print("Embeddings shape:", node_embeddings.shape)  # [num_nodes, 64]

        G_pruned = prune_graph(train_graph, node_embeddings, weight_threshold=15, sim_threshold=0.7)
        print("G pruned - Number of nodes:", G_pruned.number_of_nodes())
        print("G pruned - Number of edges:", G_pruned.number_of_edges())

        clusters, _ = calculate_clusters(G_pruned, [G_pruned.get_product_to_index(node) for node in selected_nodes], args.mode)

        #tensor_data = torch.tensor(clusters, dtype=torch.float32)
        #torch.save(tensor_data, "../data/pagerank_features.pt")
        #exit(0)
        #top_15_clusters = [[x for x in clusters[i][1:16]] for i in range(len(clusters))]
        top_15_clusters = defaultdict(set)
        for i, node in enumerate(selected_nodes):
            non_adjacent = extract_non_adjacent_nodes(G_pruned, clusters[i], G_pruned.get_product_to_index(selected_nodes[i]))
            print("len non adiacenti:", len(non_adjacent))
            if len(non_adjacent) == 15:
                clusters[i] = non_adjacent
            top_15_clusters[node] = set([G_pruned.get_index_to_product(x) for x in clusters[i]])
            print("ground truth", ground_truth[node])
            print("top 15", top_15_clusters[node])
        intersections_clusters = {key: ground_truth[key] & top_15_clusters.get(key, set()) for key in set(ground_truth) | set(top_15_clusters)}
        non_empty_count_clusters = sum(1 for intersection_set in intersections_clusters.values() if intersection_set)
        print(f"Number of nodes with non-empty intersection using clusters: {non_empty_count_clusters} out of {len(selected_nodes)}")


if __name__ == "__main__":
    main()