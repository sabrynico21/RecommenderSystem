import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.utils import negative_sampling
import numpy as np
from sklearn.metrics import roc_auc_score
from graph_utils import create_pyg_data_from_networkx

def train_lightgcn(train_graph):
    data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
    edge_index = data.edge_index
    edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
    data.num_nodes = train_graph.number_of_nodes()

    # 3. Initialize and train the model
    model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)

    if edge_weight is not None:
        history = model.train_model(edge_index, edge_weight, epochs=50)
        print("Edge weights used for training")
    else:
        history = model.train_model(edge_index, epochs=30)  # No edge weights
    torch.save(model.state_dict(), "../LightGCN/lightgcn_model.pth")
    node_embeddings = model.get_embeddings(data.edge_index, edge_weight=edge_weight)
    return node_embeddings

class LightGCN(nn.Module):
    
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Layer for initial node embeddings
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # List of Light Graph Convolution (LGConv) layers
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        
        # Initialize embeddings
        self.reset_parameters()
        
    def reset_parameters(self):
        # Standard initialization for embeddings
        nn.init.normal_(self.embedding.weight, std=0.1)
        
    def forward(self, edge_index, edge_weight=None):
        # 1. Get initial embeddings for all nodes
        x = self.embedding.weight
        
        # 2. List to store embeddings from each layer
        embeddings = [x]

        # 3. Propagate through each layer
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            # LightGCN doesn't use non-linearity between layers
            embeddings.append(x)

        # 4. Combine all layers: calculate the mean across all layers
        out = torch.stack(embeddings, dim=0).mean(dim=0)
        
        return out

    def encode(self, edge_index, edge_weight=None):
        return self.forward(edge_index, edge_weight)
    
    def decode(self, z, edge_index):
        # Simple dot product decoder
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)
    
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
                   neg_samples=1, eval_every=10, val_edges=None):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        history = {'train_loss': [], 'val_auc': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 1. Get node embeddings
            z = self.forward(edge_index, edge_weight)
            
            # 2. Sample negative edges
            neg_edge_index = negative_sampling(
                edge_index, 
                num_nodes=self.num_nodes,
                num_neg_samples=edge_index.size(1) * neg_samples
            )
            
            # 3. Compute BPR loss
            loss = self.bpr_loss(z, edge_index, neg_edge_index)
            
            # 4. Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Add this line
            optimizer.step()
            
            history['train_loss'].append(loss.item())
            
            # 5. Optional: Evaluate on validation set
            if val_edges is not None and epoch % eval_every == 0:
                pos_val_edge_index, neg_val_edge_index = val_edges
                auc_score = self.evaluate(pos_val_edge_index, neg_val_edge_index, 
                                        edge_index, edge_weight)
                history['val_auc'].append(auc_score)
                
                if epoch % (eval_every * 5) == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}, Val AUC: {auc_score:.4f}')
            else:
                if epoch % 2 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
        
        return history
    
    def evaluate(self, pos_edge_index, neg_edge_index, full_edge_index, edge_weight=None):
        self.eval()
        with torch.no_grad():
            z = self.forward(full_edge_index, edge_weight)
            
            pos_scores = self.decode(z, pos_edge_index)
            neg_scores = self.decode(z, neg_edge_index)
            
            # Combine scores and labels for AUC calculation
            scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            labels = torch.cat([torch.ones(pos_scores.size(0)), 
                              torch.zeros(neg_scores.size(0))]).cpu().numpy()
            
            auc_score = roc_auc_score(labels, scores)
            
        return auc_score
    
    def get_embeddings(self, edge_index, edge_weight=None):
        self.eval()
        with torch.no_grad():
            return self.forward(edge_index, edge_weight)