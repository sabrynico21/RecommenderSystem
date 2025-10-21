from typing import Optional, Union
from xml.parsers.expat import model

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index

from torch_geometric.utils import negative_sampling
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
device = "cpu"
from graph_utils import create_pyg_data_from_networkx
import networkx as nx


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

def train_lightgcn(train_graph, val_graph):
    data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
    data.num_nodes = train_graph.number_of_nodes()

    val_graph_remapped = remap_val_graph_to_train_ids(val_graph, val_graph.index_to_product, train_graph.product_to_index)

    val_data = create_pyg_data_from_networkx(val_graph_remapped, weight_attr='weight')
    val_edge_index = val_data.edge_index.to(device)

    # 3. Initialize and train the model
    model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3).to(device) 

    if edge_weight is not None:
        history = model.train_model(edge_index, train_graph.node_categories, edge_weight=edge_weight, val_edge_index=val_edge_index, epochs=50)
        print("Edge weights used for training")
    else:
        history = model.train_model(edge_index, train_graph.node_categories, val_edge_index=val_edge_index, epochs=30)  # No edge weights
    #torch.save(model.state_dict(), "../LightGCN/lightgcn_model.pth")
    torch.save(history, "../LightGCN/lightgcn_history_pruning.pth")
    node_embeddings = model.get_embeddings(data.edge_index, edge_weight=edge_weight)
    return node_embeddings

class LightGCN(torch.nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    :class:`~torch_geometric.nn.models.LightGCN` learns embeddings by linearly
    propagating them on the underlying graph, and uses the weighted sum of the
    embeddings learned at all layers as the final embedding

    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{(l)}_i,

    where each layer's embedding is computed as

    .. math::
        \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.

    Two prediction heads and training objectives are provided:
    **link prediction** (via
    :meth:`~torch_geometric.nn.models.LightGCN.link_pred_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.predict_link`) and
    **recommendation** (via
    :meth:`~torch_geometric.nn.models.LightGCN.recommendation_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.recommend`).

    .. note::

        Embeddings are propagated according to the graph connectivity specified
        by :obj:`edge_index` while rankings or link probabilities are computed
        according to the edges specified by :obj:`edge_label_index`.

    .. note::

        For an example of using :class:`LightGCN`, see `examples/lightgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        lightgcn.py>`_.

    Args:
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int): The dimensionality of node embeddings.
        num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        alpha (float or torch.Tensor, optional): The scalar or vector
            specifying the re-weighting coefficients for aggregating the final
            embedding. If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`~torch_geometric.nn.conv.LGConv` layers.
    """
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()


    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Computes rankings for pairs of nodes.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_label_index (torch.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
        """
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)

    def predict_link(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        prob: bool = False,
    ) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_label_index (torch.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
            prob (bool, optional): Whether probabilities should be returned.
                (default: :obj:`False`)
        """
        pred = self(edge_index, edge_label_index, edge_weight).sigmoid()
        return pred if prob else pred.round()

    def recommend(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
        sorted: bool = True,
    ) -> Tensor:
        r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
            src_index (torch.Tensor, optional): Node indices for which
                recommendations should be generated.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            dst_index (torch.Tensor, optional): Node indices which represent
                the possible recommendation choices.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            k (int, optional): Number of recommendations. (default: :obj:`1`)
            sorted (bool, optional): Whether to sort the recommendations
                by score. (default: :obj:`True`)
        """
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1, sorted=sorted).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index


    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.

        Args:
            pred (torch.Tensor): The predictions.
            edge_label (torch.Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))


    def recommendation_loss(
        self,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        node_id: Optional[Tensor] = None,
        lambda_reg: float = 1e-4,
        **kwargs,
    ) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`pos_edge_rank` vector and i-th entry
            in the :obj:`neg_edge_rank` entry must correspond to ranks of
            positive and negative edges of the same entity (*e.g.*, user).

        Args:
            pos_edge_rank (torch.Tensor): Positive edge rankings.
            neg_edge_rank (torch.Tensor): Negative edge rankings.
            node_id (torch.Tensor): The indices of the nodes involved for
                deriving a prediction for both positive and negative edges.
                If set to :obj:`None`, all nodes will be used.
            lambda_reg (int, optional): The :math:`L_2` regularization strength
                of the Bayesian Personalized Ranking (BPR) loss.
                (default: :obj:`1e-4`)
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
                function.
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        emb = self.embedding.weight
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')

    def prune_with_predict_link(self, edge_index, threshold=0.5):
        edge_probs = self.predict_link(
            edge_index=edge_index,
            edge_label_index=edge_index,
            prob=True
        )
        print('edge_probs', edge_probs)
        remove_mask = edge_probs <= threshold
        edges_to_remove = [
            (edge_index[0, i].item(), edge_index[1, i].item())
            for i in range(edge_index.size(1))
            if remove_mask[i]
        ]
        
        return edges_to_remove
    
    def decode(self, z, edge_index):
        # Simple dot product decoder
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=0)
    
    def bpr_loss(self, z, pos_edge_index, neg_edge_index, lambda_reg=1e-4):
        """
        Computes BPR loss with L2 regularization.
        
        Args:
            z: Node embeddings (output from encoder)
            pos_edge_index: Edge indices for positive samples
            neg_edge_index: Edge indices for negative samples
            lambda_reg: L2 regularization strength (default: 1e-4)
        
        Returns:
            Total loss (BPR loss + regularization)
        """
        # Decode positive and negative edges
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        
        # Compute BPR loss using log-sigmoid for numerical stability
        diff = pos_scores - neg_scores
        bpr_loss = -F.logsigmoid(diff).mean()
        
        # Add L2 regularization on the embeddings
        # Get unique node indices involved in the edges
        if lambda_reg > 0:
            # Extract unique nodes from both positive and negative edges
            pos_nodes = torch.cat([pos_edge_index[0], pos_edge_index[1]]).unique()
            neg_nodes = torch.cat([neg_edge_index[0], neg_edge_index[1]]).unique()
            node_id = torch.cat([pos_nodes, neg_nodes]).unique()
            
            # Get embeddings for involved nodes (use original embeddings, not z)
            # If you have access to self.embedding.weight, use it; otherwise use z
            if hasattr(self, 'embedding'):
                emb = self.embedding.weight[node_id]
            else:
                # Fallback: use the encoder output z for involved nodes
                emb = z[node_id]
            
            # Compute L2 regularization normalized by batch size
            regularization = lambda_reg * emb.norm(p=2).pow(2) / pos_scores.size(0)
        else:
            regularization = 0
        
        return bpr_loss + regularization
    
    def hierarchy_contrastive_loss(self, z, node_categories, temperature=0.2, level_weights=[0.4, 0.3, 0.2, 0.1], eps=1e-8):
        N = z.size(0)
        device = z.device

        # Initialize hierarchical similarity matrix
        W = torch.zeros((N, N), device=device)

        for w, key in zip(level_weights, node_categories.keys()):
            lbls_raw = node_categories[key].to(device)
            
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
    

    def train_model(self, edge_index, node_categories, edge_weight=None, epochs=300, lr=0.01, 
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

                neg_edge_index = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=self.num_nodes,
                    num_neg_samples=edge_index.size(1) * neg_samples
                )

                # Compute scores 
                pos_edge_rank = self.forward(edge_index, edge_label_index=edge_index, edge_weight=edge_weight)
                neg_edge_rank = self.forward(edge_index, edge_label_index=neg_edge_index, edge_weight=edge_weight)

                # Compute loss 
                bpr = self.recommendation_loss(pos_edge_rank, neg_edge_rank)

                # 3️⃣ Compute losses
                #bpr = self.bpr_loss(z, edge_index, neg_edge_index)
                z = self.get_embedding(edge_index, edge_weight)
                hierarchy = self.hierarchy_contrastive_loss(z, node_categories)
                #department = reparto_triplet_loss_batch(z, self.node_department) if self.node_department is not None else torch.tensor(0.0, device=device)
                #supplier = supplier_triplet_loss_batch(z, self.node_supplier) if self.node_supplier is not None else torch.tensor(0.0, device=device)
                
                # Check for NaN in losses
                if torch.isnan(bpr) or torch.isnan(hierarchy):
                    print(f"Warning: NaN detected in losses at epoch {epoch}")
                    bpr = torch.nan_to_num(bpr, nan=0.0)
                    hierarchy = torch.nan_to_num(hierarchy, nan=0.0)
                
                #loss = bpr + 0.35 * hierarchy + 0.25 * department + 0.15 * supplier
                loss = bpr + 0.3 * hierarchy

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
            z = self.get_embedding(full_edge_index, edge_weight)
            z = F.normalize(z, dim=1)

            pos_scores = self.decode(z, pos_edge_index)
            neg_scores = self.decode(z, neg_edge_index)
            
            pos_scores = torch.nan_to_num(pos_scores, nan=0.0)
            neg_scores = torch.nan_to_num(neg_scores, nan=0.0)

            scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            labels = torch.cat([
                torch.ones(pos_scores.size(0)),
                torch.zeros(neg_scores.size(0))
            ]).cpu().numpy()

            if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                print("Warning: Invalid scores detected, returning 0.5")
                return 0.5

            try:
                auc_score = roc_auc_score(labels, scores)
            except ValueError:
                print("ROC AUC calculation failed, returning 0.5")
                return 0.5

            return float(auc_score)


class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        log_prob = F.logsigmoid(positives - negatives).mean()

        regularization = 0
        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)
            regularization = regularization / positives.size(0)

        return -log_prob + regularization
