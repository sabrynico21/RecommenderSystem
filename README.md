# Random-walk-based Recommender System

This repository contains the code and experiments for the paper **"Random-walk-based Recommender System"**, which proposes a novel approach to recommendation using random walks on graphs.

The paper introduces a graph-based method that leverages the topology of item co-occurrence for personalized recommendations, aiming to demonstrate how random walks can be used to uncover item-item relationships.

## 🧠 Overview of the Method

- We model the recommendation problem as a graph, where nodes represent products, and an edge connects two products if they appear together in at least one receipt.
- Our approach relies on a modified version of the PageRank-Nibble algorithm. Starting from a given product node, it computes an Approximate Personalized PageRank vector to sample and rank related products.

## 📁 Repository Structure

