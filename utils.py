import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

def approximate_personalized_page_rank(graph, personalization_vector, beta, epsilon, mode):
    q = personalization_vector.copy()
    r = np.zeros(len(graph))
    if mode == "unweighted":
        dg = np.array([val for (node, val) in graph.degree()], dtype=float)
    else:
        dg = np.array([val for (node, val) in graph.degree(weight='weight')], dtype=float)
    dg[dg == 0] = np.inf
    while len(np.where(np.array(q)/dg > epsilon)[0]) > 0: 
        u = random.choice(np.where((np.array(q)/dg) > epsilon)[0])
        
        # Push
        r1 = r.copy()
        q1 = q.copy()
        
        r1[u] += (1 - beta) * q[u]
        q1[u] = 0.5 * beta * q[u]
        
        neighbors = list(graph.neighbors(u))
        weights = np.array([graph[u][v].get('weight', 1) for v in neighbors]) 
        sum_weights = np.sum(weights)
        
        for i, v in enumerate(neighbors):
            if mode == "unweighted":
                q1[v] += 0.5 * beta * (q[u]/ len(neighbors))
            else:
                q1[v] += 0.5 * beta * (q[u] * weights[i] / sum_weights)
        
        r = r1 / np.sum(r1)
        q = q1
    
    return np.vstack((r, q))

# def calculate_cut(graph, cut, x, pr):
#     deg = graph.degree(pr[x], weight='weight')
#     if x == 0:
#         return deg
#     else:
#         te = x - 1
#         e = sum(1 for neighbor in pr[:te] if graph.has_edge(pr[x], neighbor))
#         return cut[pr[x-1]] + deg - 2 * e

def calculate_cut(graph, cut, x, pr, weighted=True):
    if weighted:
        deg = graph.degree(pr[x], weight='weight')
    else:
        deg = graph.degree(pr[x])
    if x == 0:
        return deg
    else:
        te = x - 1
        if weighted:
            e = sum(graph[pr[x]][neighbor]['weight'] for neighbor in pr[:te] if graph.has_edge(pr[x], neighbor))  # Count weighted edges
        else:
            e = sum(1 for neighbor in pr[:te] if graph.has_edge(pr[x], neighbor))  # Count unweighted edges
        return cut[pr[x-1]] + deg - 2 * e

# def calculate_conductance(graph, pr, supp):
#     n = len(graph)
#     cut = np.zeros(n)
#     vol = np.zeros(n)
#     x = 0
    
#     degrees = dict(graph.degree(pr, weight='weight'))
#     vol[pr] = np.cumsum([degrees[node] for node in pr])
    
#     range_nodes = pr[:supp]
    
#     for i in range_nodes:
#         cut[i] = calculate_cut(graph, cut, x, pr)
#         x += 1
    
#     conductance = cut / vol
#     return np.vstack((conductance, vol))

def calculate_conductance(graph, pr, supp, weighted=True):
    n = len(graph)
    cut = np.zeros(n)
    vol = np.zeros(n)
    x = 0
    if weighted:
        degrees = dict(graph.degree(pr, weight='weight'))
    else:
        degrees = dict(graph.degree(pr))  # Ignore weights
    vol[pr] = np.cumsum([degrees[node] for node in pr])
    range_nodes = pr[:supp]
    for i in range_nodes:
        cut[i] = calculate_cut(graph, cut, x, pr, weighted) 
        x += 1
    conductance = cut / vol
    return np.vstack((conductance, vol))

def page_rank_nibble(graph, n, phi, beta, epsilon, mode, seed):
    personalization_vector = np.zeros(n)
    if seed == -1:
        x = random.choice(range(n))
    else:
        x = seed
    if mode=="unweighted":
        personalization_vector[x] = 1
    else:
        neighbors = list(graph.neighbors(x))
        weights = np.array([graph[x][v].get('weight', 1) for v in neighbors]) 
        sum_weights = np.sum(weights)
        personalization_vector[x] = sum_weights
    
    m = sum(dict(graph.degree()).values()) / 2
    B = int(np.floor(np.log(m)))
    b = random.choice(range(1, B+1))

    rq = approximate_personalized_page_rank(graph, personalization_vector, beta, epsilon, mode)
    
    #print("rq:", rq)
    r = rq[0, :]
    
    r_s = np.argsort(-r)
    #print("r_s:", r_s)
    supp = np.sum(r > 0)
    #print("sup", supp)
    covo = calculate_conductance(graph, r_s, supp, True if mode=="weighted" else False)
    
    supp = np.sum(r > 0)
    #print("sup", supp)
    co = covo[0, :]
    #print(co[r_s])
    # plt.plot(range(0, n), co[r_s])
    # plt.xlabel('Node')
    # plt.ylabel('Conductance')
    #plt.show()
    
    cluster = 0
    for i in range(supp):
        if r[r_s[i]] > 0:
            if covo[1, r_s[i]] > 2**(b-1):
                cluster = i
    
    return (x,r_s[:cluster+1])