# Random-walk-based Recommender System

This repository contains the code and experiments for the paper **"Random-walk-based Recommender System"**, which proposes a novel approach to recommendation using random walks on graphs.

The paper introduces a graph-based method that leverages the topology of item co-occurrence for personalized recommendations, aiming to demonstrate how random walks can be used to uncover item-item relationships.

## 🧠 Overview of the Method

- We model the recommendation problem as a graph, where nodes represent products, and an edge connects two products if they appear together in at least one receipt.
- Our approach relies on a modified version of the PageRank-Nibble algorithm. Starting from a given product node, it computes an Approximate Personalized PageRank vector to sample and rank related products.

## 📁 Repository Structure

.
├── data/                        # Graph data used in experiments
│   └── *.pkl                    # Pickled NetworkX graphs
│                                # Generated or loaded depending on script args
│
├── src/                         # Source code
│   ├── validation.py            # Main script to run experiments
│                                # Args:
│                                #   --load (bool): Load graph from .pkl or build new
│                                #   --mode (str): 'weighted' or 'unweighted'
│                                #   --t_min / --t_max (float): Edge weight thresholds
│                                #   --task (str): 'predictions', 'performances', 'frequent_items'
│   ├── display_plots.py         # Script for showing result plots
│
├── results/                     # Output from experiments
│   ├── Link prediction/         # Visuals and metrics for link prediction task
│   ├── Frequent Itemset Comparison/
│                                # Comparison with frequent itemset mining methods
│   ├── w_test_epsilon_performances.pkl
│   └── unw_test_epsilon_performances.pkl
│                                # Metrics: CCR, F1-score, execution time (weighted/unweighted)
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation


