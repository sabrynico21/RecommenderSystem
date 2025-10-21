import numpy as np
import random
from collections import defaultdict
random.seed(42)

# def approximate_personalized_page_rank(graph, q, beta, epsilon, mode):
#     r = defaultdict(float)
    
#     dg = graph.get_deg(weight=None if mode == "unweighted" else 'weight')
#     iterations= 0
#     #prev_top_list = None
#     prev_r = None
#     retention_ratio = 0.5
#     contrib = (1 - retention_ratio) * beta
#     MAX_NEIGHBORS = 100 
#     while True:
#         if iterations > 5000:
#            print('Max iterations reached')
#            break
        
#         candidates = [u for u in q if q[u] / (dg[u] if dg[u] != 0 else float('inf')) > epsilon]
#         #print(" candidates:", len(candidates))
#         if not candidates:
#             print("nessun candidato")
#             epsilon = epsilon / 1.1
#             continue
#         u = random.choice(candidates)
#         #u = get_priority_candidate(candidates, dg, q)
    
#         push_val = q[u]
        
#         r[u] += (1 - beta) * push_val
#         q[u] = retention_ratio * beta * push_val

#         # if contrib * push_val < 0.000001 * min(graph.get_deg(u), MAX_NEIGHBORS):
#         #     continue

#         neighbors = list(graph.get_neighbors(u))
#         if not neighbors:
#             continue
#         iterations +=1
        
#         if len(neighbors) > MAX_NEIGHBORS:
#             neighbors = random.sample(neighbors, MAX_NEIGHBORS)

#         if mode == "unweighted":
#             for v in neighbors:
#                 q[v] += contrib * push_val / len(neighbors)
#         else:
#             weights = [graph.get_weight(u, v) for v in neighbors]
#             #print("weights", weights)
#             sum_weights = sum(weights)
#             for i, v in enumerate(neighbors):
#                 q[v] += contrib * push_val * weights[i] / sum_weights

#         # if iterations % 200 == 0 and len(r) >= 30:
#         #     # normalize by degree (no need to divide by total for ranking)
#         #     r_norm = {node: r[node] / dg[node] for node in r}
#         #     import heapq
#         #     # get top-30 node IDs efficiently
#         #     top_items = heapq.nlargest(30, r_norm.items(), key=lambda x: x[1])
#         #     top_list = [node for node, _ in top_items]

#         #     if prev_top_list is not None:
#         #         common = len(set(top_list) & set(prev_top_list))
#         #         coverage = common / len(top_list)
#         #         if coverage >= 0.9:  # 90% overlap
#         #             break

#         #     prev_top_list = top_list    
#         if iterations % 300 == 0:
#             # Normalize r by degree for fair comparison
#             r_norm = {node: r[node] / dg[node] for node in r}
#             # Convert to a vector for comparison
#             r_vec = np.array([r_norm.get(node, 0.0) for node in dg])
#             if prev_r is not None:
#                 delta = np.sum(np.abs(r_vec - prev_r))

#                 if delta < 0.01: 
#                     print(f"Converged at iteration {iterations} with delta={delta:.2e}")
#                     break
#             prev_r = r_vec.copy()
#     # Normalize r by degree
#     r = {node: r[node] / dg[node] for node in r}
#     total = sum(r.values())
#     r = {node: val / total for node, val in r.items()}
    
#     return r

from collections import defaultdict
import random
import numpy as np

from collections import defaultdict
import random
import numpy as np
import heapq

from collections import defaultdict, deque
import random
import numpy as np

def approximate_personalized_page_rank(graph, q, beta, epsilon, mode):
    r = defaultdict(float)
    dg = graph.get_deg(weight=None if mode == "unweighted" else 'weight')
    iterations = 0
    prev_r = None

    retention_ratio = 0.5
    contrib = (1 - retention_ratio) * beta
    MAX_NEIGHBORS = 50 

    # --- Weight cache ---
    weight_cache = {} 
    MAX_CACHE_SIZE = 1000000
    PRUNE_RATIO = 0.5

    # --- Neighbor cache ---
    neighbor_cache = {}
    MAX_NEIGHBOR_CACHE = 100000
    NEIGHBOR_PRUNE_RATIO = 0.5

    # --- Candidate queue ---
    candidate_queue = deque()

    # -------------------------------
    # Caching helpers
    # -------------------------------

    def get_cached_weight(u, v):
        """Retrieve edge weight from cache or graph, caching as needed."""
        key = tuple(sorted((u, v)))  # undirected graph
        if key in weight_cache:
            return weight_cache[key]
        w = graph.get_weight(u, v)
        weight_cache[key] = w
        if len(weight_cache) > MAX_CACHE_SIZE:
            prune_weight_cache()
        return w

    def prune_weight_cache():
        """Reduce cache size to limit memory usage."""
        nonlocal weight_cache
        keep_n = int(MAX_CACHE_SIZE * PRUNE_RATIO)
        sampled_items = random.sample(list(weight_cache.items()), keep_n)
        weight_cache = dict(sampled_items)
        print(f"[Weight cache pruned] New size: {len(weight_cache)}")

    def get_cached_neighbors(u):
        """Retrieve neighbor list from cache or graph."""
        if u in neighbor_cache:
            return neighbor_cache[u]
        nbs = list(graph.get_neighbors(u))
        neighbor_cache[u] = nbs
        if len(neighbor_cache) > MAX_NEIGHBOR_CACHE:
            prune_neighbor_cache()
        return nbs

    def prune_neighbor_cache():
        """Reduce neighbor cache to limit memory usage."""
        nonlocal neighbor_cache
        keep_n = int(MAX_NEIGHBOR_CACHE * NEIGHBOR_PRUNE_RATIO)
        sampled_items = random.sample(list(neighbor_cache.items()), keep_n)
        neighbor_cache = dict(sampled_items)
        print(f"[Neighbor cache pruned] New size: {len(neighbor_cache)}")

    # -------------------------------
    # Main loop
    # -------------------------------
    while True:
        if iterations > 10000:
            print('Max iterations reached')
            break
        
        # Refill queue if empty
        if not candidate_queue:
            candidates = [u for u in q if q[u] / (dg[u] if dg[u] != 0 else float('inf')) > epsilon]
            if not candidates:
                print("No candidates found, reducing epsilon")
                epsilon /= 1.1
                continue
            candidate_queue.extend(candidates)

        u = candidate_queue.popleft()
        push_val = q[u]
        scale = contrib * push_val

        # Update r and q[u]
        r[u] += (1 - beta) * push_val
        q[u] = retention_ratio * beta * push_val

        # if scale < epsilon * min(dg[u], MAX_NEIGHBORS):
        #     continue

        # ✅ Cached neighbor retrieval
        neighbors = get_cached_neighbors(u)
        if not neighbors:
            continue
        iterations += 1

        if len(neighbors) > MAX_NEIGHBORS:
            neighbors = random.sample(neighbors, MAX_NEIGHBORS)

        if mode == "unweighted":
            inc = scale / len(neighbors)
            for v in neighbors:
                q[v] += inc
        else:
            weights = [get_cached_weight(u, v) for v in neighbors]
            sum_weights = sum(weights)
            scale_norm = scale / sum_weights
            for i, v in enumerate(neighbors):
                q[v] += scale_norm * weights[i]

        # --- Convergence check ---
        if iterations % 500 == 0:
            r_norm = {node: r[node] / dg[node] for node in r}
            r_vec = np.array([r_norm.get(node, 0.0) for node in dg])
            if prev_r is not None:
                delta = np.sum(np.abs(r_vec - prev_r))
                rel_delta = delta / (np.sum(np.abs(prev_r)) + 1e-12)
                if rel_delta < 0.05:
                    print(f"Converged at iteration {iterations} (rel_delta={rel_delta:.2e})")
                    break
            prev_r = r_vec.copy()

    # --- Normalize final result ---
    r = {node: r[node] / dg[node] for node in r}
    total = sum(r.values()) 
    r = {node: val / total for node, val in r.items()}
    
    return r


def get_priority_candidate(candidates, dg, q):
    """Select candidate with probability proportional to sqrt(degree) * residual"""
    if not candidates:
        return None
    
    # Create weights: balance between residual and degree importance
    weights = []
    for u in candidates:
        residual_norm = q[u] / max(dg[u], 1.0)
        degree_weight = min(10.0, max(1.0, dg[u] ** 0.5))  # sqrt scaling
        weight = residual_norm * degree_weight
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        return random.choice(candidates)
    
    probabilities = [w / total_weight for w in weights]
    return random.choices(candidates, weights=probabilities)[0]


def compute_epsilon(graph, x, c):
    print(c)
    degree = graph.get_deg(nodes=x, weight= None)
    print("degree: ", degree)
    n = graph.number_of_nodes()
    m = graph.number_of_edges() 
    epsilon = c * (n / m) * (1 / np.sqrt(degree + 1))
    #epsilon = c * (n / m)
    return epsilon

def page_rank_nibble(graph, beta, mode, seed, c):
    n = graph.number_of_nodes()
    personalization_dict = defaultdict(float)
    if seed == -1:
        seed_node = random.choice(range(n))
    else:
        if not graph.has_node(seed):
            return (seed, [])
        seed_node = seed
    if mode == "unweighted":
        personalization_dict[seed_node] = 1
    else:
        degree = graph.get_deg(nodes=seed_node, weight="weight")
        personalization_dict[seed_node] = degree
    
        
    epsilon = compute_epsilon(graph, seed_node, c)
    print("epsilon:", epsilon)
    r = approximate_personalized_page_rank(graph, personalization_dict, beta, epsilon, mode)
    
    sorted_r_keys = [k for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)]
    #print("len r:" ,len(r))
    print("len sorted r:" ,len(sorted_r_keys))
    return (seed_node, sorted_r_keys)