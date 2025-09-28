import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.utils import negative_sampling
import numpy as np
from sklearn.metrics import roc_auc_score
from graph_utils import create_pyg_data_from_networkx
import networkx as nx
import os
from torch.amp import GradScaler, autocast
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def remap_val_graph_to_train_ids(val_graph, val_to_product_map, product_to_train_map, drop_missing=True):
    mapping = {}
    missing = []
    
    for val_node_id in val_graph.nodes():
        product_id = val_to_product_map[val_node_id]
        if product_id in product_to_train_map:
            mapping[val_node_id] = product_to_train_map[product_id]
        else:
            missing.append((val_node_id, product_id))
    
    if missing and not drop_missing:
        raise KeyError(f"Missing product_ids in train map: {missing[:10]}{'...' if len(missing)>10 else ''}")
    
    # Relabel the validation graph, dropping missing nodes if necessary
    val_graph_remapped = nx.relabel_nodes(val_graph.graph, mapping, copy=True)
    if drop_missing and missing:
        print("nodi rimossi", len(missing))
        val_graph_remapped.remove_nodes_from([n for n, _ in missing])
    
    return val_graph_remapped


def train_lightgcn(train_graph, node_labels, val_graph):
    data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
    data.num_nodes = train_graph.number_of_nodes()

    val_graph_remapped = remap_val_graph_to_train_ids(val_graph, val_graph.index_to_product, train_graph.product_to_index)

    val_data = create_pyg_data_from_networkx(val_graph_remapped, weight_attr='weight')
    val_edge_index = val_data.edge_index.to(device)

    # 3. Initialize and train the model
    model = LightGCN(num_nodes=data.num_nodes, node_labels=node_labels, embedding_dim=64, num_layers=3).to(device)

    if edge_weight is not None:
        history = model.train_model(edge_index, edge_weight, val_edge_index=val_edge_index, epochs=100)
        print("Edge weights used for training")
    else:
        history = model.train_model(edge_index, epochs=30)  # No edge weights
    #torch.save(model.state_dict(), "../LightGCN/lightgcn_model.pth")
    torch.save(history, "../LightGCN/lightgcn_history_pruning.pth")
    node_embeddings = model.get_embeddings(data.edge_index, edge_weight=edge_weight)
    return node_embeddings


def supplier_triplet_loss_batch(z, supplier_labels, margin=0.5, num_triplets=1000):
    """
    Batch-optimized version for better GPU utilization
    """
    device = z.device
    z = F.normalize(z, p=2, dim=1)
    
    # Create triplets in batch
    anchors, positives, negatives = [], [], []
    
    supplier_to_indices = {}
    for idx, supplier in enumerate(supplier_labels):
        supplier = supplier.item() if torch.is_tensor(supplier) else supplier
        if supplier not in supplier_to_indices:
            supplier_to_indices[supplier] = []
        supplier_to_indices[supplier].append(idx)
    
    # Generate triplets
    for _ in range(num_triplets):
        # Random supplier
        supplier = random.choice(list(supplier_to_indices.keys()))
        indices = supplier_to_indices[supplier]
        
        if len(indices) < 2:
            continue
            
        # Random anchor and positive from same supplier
        anchor_idx, positive_idx = random.sample(indices, 2)
        
        # Random negative from different supplier
        other_suppliers = [s for s in supplier_to_indices.keys() if s != supplier]
        if not other_suppliers:
            continue
        other_supplier = random.choice(other_suppliers)
        negative_idx = random.choice(supplier_to_indices[other_supplier])
        
        anchors.append(anchor_idx)
        positives.append(positive_idx)
        negatives.append(negative_idx)
    
    if not anchors:
        return torch.tensor(0.0, device=device)
    
    # Batch distance computation
    anchor_emb = z[anchors]
    positive_emb = z[positives]
    negative_emb = z[negatives]
    
    pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
    neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
    
    loss = F.relu(pos_dist - neg_dist + margin).mean()
    return loss

import random

def reparto_triplet_loss_batch(z, reparto_labels, margin=0.8, num_triplets=1000):
    """Batch version for reparto"""
    device = z.device
    z = F.normalize(z, p=2, dim=1)
    
    anchors, positives, negatives = [], [], []
    
    reparto_to_indices = {}
    for idx, reparto in enumerate(reparto_labels):
        reparto = reparto.item() if torch.is_tensor(reparto) else reparto
        if reparto not in reparto_to_indices:
            reparto_to_indices[reparto] = []
        reparto_to_indices[reparto].append(idx)
    
    for _ in range(num_triplets):
        reparto = random.choice(list(reparto_to_indices.keys()))
        indices = reparto_to_indices[reparto]
        
        if len(indices) < 2:
            continue
            
        anchor_idx, positive_idx = random.sample(indices, 2)
        
        other_repartos = [r for r in reparto_to_indices.keys() if r != reparto]
        if not other_repartos:
            continue
        other_reparto = random.choice(other_repartos)
        negative_idx = random.choice(reparto_to_indices[other_reparto])
        
        anchors.append(anchor_idx)
        positives.append(positive_idx)
        negatives.append(negative_idx)
    
    if not anchors:
        return torch.tensor(0.0, device=device)
    
    anchor_emb = z[anchors]
    positive_emb = z[positives]
    negative_emb = z[negatives]
    
    pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
    neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
    
    return F.relu(pos_dist - neg_dist + margin).mean()

class LightGCN(nn.Module):
    def __init__(self, num_nodes, node_labels=None, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        if node_labels is not None:
            self.node_categories = {}
            for level in ["descr_liv1", "descr_liv2", "descr_liv3", "descr_liv4"]:
                if level in node_labels.columns:
                    self.node_categories[level] = torch.tensor(
                        node_labels[level].values, 
                        dtype=torch.long,
                        device=device
                    )
   
            # if "descr_forn" in node_labels.columns:
            #     self.node_supplier = torch.tensor(
            #         node_labels["descr_forn"].values,
            #         dtype=torch.long,
            #         device=device
            #     )

            # if "descr_rep" in node_labels.columns:
            #     self.node_department = torch.tensor(
            #         node_labels["descr_rep"].values,
            #         dtype=torch.long,
            #         device=device
            #     )
        
        self.embedding = nn.Embedding(num_nodes, embedding_dim).to(device)
        self.convs = nn.ModuleList([LGConv().to(device) for _ in range(num_layers)])
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

    def encode(self, edge_index, edge_weight=None):
        return self.forward(edge_index, edge_weight)
    
    def decode(self, z, edge_index):
        # Simple dot product decoder
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)
        
    def forward(self, edge_index, edge_weight=None):
        x = self.embedding.weight
        embeddings = [x]
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            embeddings.append(x)
        
        return torch.stack(embeddings, dim=0).mean(dim=0)

    import torch.nn.functional as F

    def hierarchy_contrastive_loss(self, z, temperature=0.2, level_weights=[0.4, 0.3, 0.2, 0.1], eps=1e-8):
        N = z.size(0)
        device = z.device

        # Initialize hierarchical similarity matrix
        W = torch.zeros((N, N), device=device)

        for w, key in zip(level_weights, self.node_categories.keys()):
            lbls_raw = self.node_categories[key].to(device)
            
            # Handle label alignment safely
            lbls = torch.full((N,), -1, dtype=torch.long, device=device)
            num_valid = min(len(lbls_raw), N)
            lbls[:num_valid] = lbls_raw[:num_valid]

            # Only keep nodes with valid labels
            valid_mask = (lbls != -1)
            if valid_mask.sum() < 2:  # Need at least 2 nodes for comparison
                continue

            valid_indices = torch.where(valid_mask)[0]
            lbls_valid = lbls[valid_indices]
            
            # Compute similarity for valid nodes
            same = (lbls_valid.unsqueeze(0) == lbls_valid.unsqueeze(1)).float()
            
            # Add weighted similarity to W
            W[valid_indices.unsqueeze(1), valid_indices] += w * same

        # Normalize embeddings safely
        z = F.normalize(z, dim=1)
        z = torch.nan_to_num(z, nan=0.0)

        # Cosine similarity matrix
        sim = torch.matmul(z, z.T) / temperature
        sim = torch.nan_to_num(sim, nan=-1e9)  # Replace NaN with very negative value

        # Mask self-similarity
        mask_eye = torch.eye(N, device=device).bool()
        sim = sim.masked_fill(mask_eye, -1e9)

        # Exponentiate similarities
        exp_sim = torch.exp(sim)
        exp_sim = torch.nan_to_num(exp_sim, nan=0.0)

        # Denominator and numerator with clamping
        denom = exp_sim.sum(dim=1, keepdim=True).squeeze()
        denom = denom.clamp(min=eps)

        numer = (exp_sim * W).sum(dim=1)
        numer = numer.clamp(min=eps)

        # Only compute loss for nodes with positives
        valid_nodes = (W.sum(dim=1) > 0)
        if not valid_nodes.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = torch.zeros(N, device=device)
        loss[valid_nodes] = -torch.log(numer[valid_nodes] / denom[valid_nodes])

        return loss.mean()


    
    def bpr_loss(self, z, pos_edge_index, neg_edge_index):
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        
        # Use log-sigmoid for better numerical stability
        # Instead of: -log(sigmoid(pos_score - neg_score))
        # Use: -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Even more stable: compute difference first
        diff = pos_scores - neg_scores
        loss = -F.logsigmoid(diff).mean()
        
        # Add a small epsilon to prevent log(0)
        # loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
        
        return loss
    
    def train_model(self, edge_index, edge_weight=None, epochs=300, lr=0.01, 
                neg_samples=1, eval_every=10, val_edge_index=None):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        history = {'train_loss': [], 'val_auc': []}
        best_auc = 0.0

        # Setup scheduler if validation provided
        if val_edge_index is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                min_lr=1e-5
            )
            neg_val_edge_index = negative_sampling(
                val_edge_index,
                num_nodes=self.num_nodes,
                num_neg_samples=val_edge_index.size(1)
            )
            val_edges = (val_edge_index, neg_val_edge_index)
        else:
            scheduler = None
            val_edges = None
            print("No validation data - scheduler disabled")

        scaler = GradScaler()
        
        for epoch in range(epochs):
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                # 1️⃣ Node embeddings
                z = self.forward(edge_index, edge_weight)
                z = F.normalize(z, dim=1)
                z = torch.nan_to_num(z, nan=0.0)

                # 2️⃣ Sample negative edges
                neg_edge_index = negative_sampling(
                    edge_index,
                    num_nodes=self.num_nodes,
                    num_neg_samples=edge_index.size(1) * neg_samples
                )

                # 3️⃣ Compute losses
                bpr = self.bpr_loss(z, edge_index, neg_edge_index)
                hierarchy = self.hierarchy_contrastive_loss(z)
                #department = reparto_triplet_loss_batch(z, self.node_department) if self.node_department is not None else torch.tensor(0.0, device=device)
                #supplier = supplier_triplet_loss_batch(z, self.node_supplier) if self.node_supplier is not None else torch.tensor(0.0, device=device)
                
                # Check for NaN in losses
                if torch.isnan(bpr) or torch.isnan(hierarchy):
                    print(f"Warning: NaN detected in losses at epoch {epoch}")
                    bpr = torch.nan_to_num(bpr, nan=0.0)
                    hierarchy = torch.nan_to_num(hierarchy, nan=0.0)
                
                #loss = bpr + 0.35 * hierarchy + 0.25 * department + 0.15 * supplier
                loss = bpr + 0.4 * hierarchy

            # 4️⃣ Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Check for NaN gradients before unscaling
            has_nan_grad = False
            for p in self.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN gradients detected at epoch {epoch}, skipping update")
                optimizer.zero_grad()
                continue
            
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update parameters
            scaler.step(optimizer)
            scaler.update()
            
            history['train_loss'].append(loss.item())

            # 5️⃣ Logging
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch:03d} | BPR: {bpr:.4f}, Hierarchy: {hierarchy:.4f}") #, "
                    #f"Dept: {department:.4f}, Supplier: {supplier:.4f}, Total: {loss:.4f}")
                
                # Save checkpoint (optional - can be heavy)
                if (epoch + 1) % 50 == 0:  # Save less frequently
                    torch.save(self.state_dict(), f"../LightGCN/lightgcn_model_{epoch:03d}.pth")

            # 6️⃣ Validation
            if val_edges is not None and (epoch + 1) % eval_every == 0:
                val_pos, val_neg = val_edges
                auc_score = self.evaluate(val_pos, val_neg, edge_index, edge_weight)

                if np.isnan(auc_score):
                    auc_score = 0.0
                    print(f"Warning: NaN AUC at epoch {epoch}")

                history['val_auc'].append(auc_score)
                if auc_score > best_auc:
                    best_auc = auc_score
                    # Save best model
                    torch.save(self.state_dict(), f"../LightGCN/lightgcn_best_pruning_{epoch:03d}.pth")

                if scheduler is not None:
                    scheduler.step(auc_score)

                if (epoch + 1) % 10 == 0:
                    print(f"Validation AUC: {auc_score:.4f}, Best AUC: {best_auc:.4f}")
                    
            # Clear cache every few epochs
            if epoch % 20 == 0:
                torch.cuda.empty_cache()

        print(f"Training completed. Best validation AUC: {best_auc:.4f}")
        return history

    
    def evaluate(self, pos_edge_index, neg_edge_index, full_edge_index, edge_weight=None):
        self.eval()
        with torch.no_grad():
            z = self.forward(full_edge_index, edge_weight)
            
            pos_scores = self.decode(z, pos_edge_index)
            neg_scores = self.decode(z, neg_edge_index)
            
            # Add NaN protection
            pos_scores = torch.nan_to_num(pos_scores, nan=0.0)
            neg_scores = torch.nan_to_num(neg_scores, nan=0.0)
            
            # Combine scores and labels
            scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            labels = torch.cat([torch.ones(pos_scores.size(0)), 
                            torch.zeros(neg_scores.size(0))]).cpu().numpy()
            
            # Additional check for NaN in numpy arrays
            if np.any(np.isnan(scores)):
                print("Warning: NaN detected in evaluation scores, returning 0.5")
                return 0.5
            
            try:
                auc_score = roc_auc_score(labels, scores)
            except ValueError:
                print("ROC AUC calculation failed, returning 0.5")
                return 0.5
            
        return auc_score
    
    def get_embeddings(self, edge_index, edge_weight=None):
        self.eval()
        with torch.no_grad():
            # Use all layers without storing intermediate values
            x = self.embedding.weight
            out = x.clone()  # Start with initial embeddings
            
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index, edge_weight)
                out += x  # Accumulate embeddings
            
            return out / (self.num_layers + 1)  # Average