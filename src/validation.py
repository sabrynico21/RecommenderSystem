import pickle
import argparse
import clickhouse_connect
import os
from dotenv import load_dotenv
from graph_utils import *
from graph import Graph
from marketbasket import save_frequent_items

def main():

    parser = argparse.ArgumentParser(description='Insert thresholds for the graph cluster algorithm')
    parser.add_argument('--load', default= "True", help='Specify if load the graph or create a new one')
    parser.add_argument('--t_min', default=0, help='Specify the min threshold')
    parser.add_argument('--t_max', default=float('inf'), help='Specify the max threshold')
    parser.add_argument('--mode', default="weighted", choices=["weighted","unweighted"], help='Specify whether to use weighted or unweighted graph')
    parser.add_argument('--seed', help='Specify id of the seed node')
    parser.add_argument('--task', default="performances", choices=["predictions", "performances","frequent_items"], help='Specify the task to perform')
    args = parser.parse_args()
    load_dotenv()

    client = clickhouse_connect.get_client(
    host=os.getenv('CLICKHOUSE_HOST'),
    port=int(os.getenv('CLICKHOUSE_PORT')),
    username=os.getenv('CLICKHOUSE_USER'),
    password=os.getenv('CLICKHOUSE_PASSWORD'),
    database=os.getenv('CLICKHOUSE_DATABASE')
    )
    
    if args.load == "False":
        #table_name = 'dati_scontrini'
        # edge_weights = calulate_edge_weights(client, table_name)
        # with open('edge_weights.pkl', 'wb') as file:
        #     pickle.dump(edge_weights, file)
        
        with open('../data/edge_weights.pkl', 'rb') as file:
           edge_weights = pickle.load(file)

        #display_edge_weight_distribution(edge_weights)
        #fit_powerlaw_on_edge_distribution_CDDF(edge_weights)
    
        graph = Graph()
        graph.create_graph(edge_weights, int(args.t_min), float(args.t_max))
        #plot_degree_distribution(graph)
        #fit_powerlaw_on_degree_distribution_CDDF(graph)
        
        with open(f"../data/graph_{graph.t_min}-{graph.t_max}.pkl", "wb") as f:
            pickle.dump(graph.get_graph(), f)
        
        with open(f'../data/products_dict_{graph.t_min}-{graph.t_max}.pkl', 'wb') as f:
            pickle.dump({'id_to_index': graph.get_product_to_index(), 'index_to_id': graph.get_index_to_product()}, f)
        
    else:
        file_path = "../data/graph.pkl"
        graph = Graph.load_graph(file_path, int(args.t_min), float(args.t_max))
        dict_path = "../data/products_dict.pkl"
        graph.load_dicts(dict_path)    

    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    exit(0)
    
    if args.task == "predictions":
    
        #Link Prediction task
        testing_file_path = "../Results/Link prediction/prova.txt"
        weight = 50
        dev = 10
        num = 50
        random_edges = extract_random_edges(edge_weights, weight, dev, num, testing_file_path)
        with open(testing_file_path, 'a') as f:
            f.write(f"EDGE_WEIGHT: {weight - dev} - {weight + dev}\n")
            f.write(f"graph: {args.t_min} - {args.t_max}\n")
        link_prediction(random_edges, graph, client, testing_file_path, args.mode)

    elif args.task == "performances":
        # Performance evaluation with Reduced Graphs
        percent = 30
        reduced_graph = remove_random_edges(graph, percent)
        print("Number of nodes:", reduced_graph.number_of_nodes())
        print("Number of edges:", reduced_graph.number_of_edges())
        validation(graph, reduced_graph, args.mode)

    elif args.task == "frequent_items":
        # Frequent Itemsets for selected nodes
        with open("../data/selected_nodes.txt") as f:
            selected_nodes = ast.literal_eval(f.read())
        
        save_frequent_items( selected_nodes)

if __name__ == "__main__":
    main()