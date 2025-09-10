import numpy as np
import random
from collections import defaultdict

def approximate_personalized_page_rank(graph, q, beta, epsilon, mode):
    r = defaultdict(float)
    
    dg = {
        node: float(degree)
        for node, degree in graph.get_degree(weight=None if mode == "unweighted" else 'weight')
    }
    iterations= 0
    prev_top_list = None
    while True:
        #if iterations > 5000:
        #    break
        iterations +=1
        candidates = [u for u in q if q[u] / (dg[u] if dg[u]!=0 else float('inf')) > epsilon]
        #print(" candidates:", len(candidates))
        if not candidates:
            print("nessun candidato")
            epsilon = epsilon / 1.1
            continue
        u = random.choice(candidates)
        
        retention_ratio = 0.5
        push_val = q[u]
        r[u] += (1 - beta) * push_val
        q[u] = retention_ratio * beta * push_val

        neighbors = list(graph.get_neighbors(u))
        if not neighbors:
            continue
        contrib = (1 - retention_ratio) * beta

        if mode == "unweighted":
            for v in neighbors:
                q[v] += contrib * push_val / len(neighbors)
        else:
            weights = [graph.get_weight(u, v) for v in neighbors]
            #print("weights", weights)
            sum_weights = sum(weights)
            for i, v in enumerate(neighbors):
                q[v] += contrib * push_val * weights[i] / sum_weights

        if iterations % 100 == 0 and len(r) >= 30:
            # normalize by degree (no need to divide by total for ranking)
            r_norm = {node: r[node] / dg[node] for node in r}
            import heapq
            # get top-30 node IDs efficiently
            top_items = heapq.nlargest(30, r_norm.items(), key=lambda x: x[1])
            top_list = [node for node, _ in top_items]

            if prev_top_list is not None:
                matches = sum(a == b for a, b in zip(top_list, prev_top_list))
                if matches >= 24:  # i.e., >= 0.9 overlap
                    break

            prev_top_list = top_list    
    # Normalize r by degree
    r = {node: r[node] / dg[node] for node in r}
    total = sum(r.values())
    r = {node: val / total for node, val in r.items()}
    
    return r

def compute_epsilon(graph, x, mode):
    if mode == "unweighted":
        c = 0.002 
    else:
        c = 0.00006 
    
    print(c)
    degree = graph.get_degree(nodes=x)
    print("degree: ", degree)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    epsilon = c * (n / m) * (1 / np.sqrt(degree + 1))
    return epsilon

def page_rank_nibble(graph, beta, mode, seed):
    n = graph.number_of_nodes()
    personalization_dict = defaultdict(float)
    if seed == -1:
        seed_node = random.choice(range(n))
    else:
        if not graph.has_node(seed):
            return (seed, [])
        seed_node = seed
    personalization_dict[seed_node] = 1
        
    epsilon = compute_epsilon(graph, seed_node, mode)
    print("epsilon:", epsilon)
    r = approximate_personalized_page_rank(graph, personalization_dict, beta, epsilon, mode)
    
    sorted_r_keys = [k for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)]
    #print("len r:" ,len(r))
    print("len sorted r:" ,len(sorted_r_keys))
    return (seed_node, sorted_r_keys)