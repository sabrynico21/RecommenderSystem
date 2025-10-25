import inspect
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np

import pickle
import pandas as pd
import torch
from scipy.sparse import csr_matrix
import MemoryLogger
from graph import simple_align_labels
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-memory', action='store_true')
    args = parser.parse_args()
    logger = MemoryLogger.MemoryLogger(enabled=args.log_memory)

    def load_receipt_features(csv_path, receipt_ids=None, exclude_products=None):
        """
        Load a CSV file where each row is a receipt and contains a space-separated list of product IDs.
        
        Args:
            csv_path: Path to the CSV file
            receipt_ids: Optional list of receipt IDs to include. If None, uses all receipts.
                        The output tensor will always have columns for all receipts in the CSV.
        
        Returns:
            features_sparse_tensor: torch.sparse.FloatTensor [num_products, num_all_receipts]
            product_to_idx: dict mapping product ID -> row index
            idx_to_product: dict mapping row index -> product ID
            all_receipt_ids: list of all receipt IDs in the CSV (in original order)
        """
        # Load CSV
        logger.log_operation("load", "Loading CSV file", __file__, inspect.currentframe().f_lineno)
        df = pd.read_csv(csv_path, header=0, names=["receipt_id", "products"])
        logger.log_operation("load", "CSV file loaded", __file__, inspect.currentframe().f_lineno, size=df.memory_usage(deep=True).sum())

        # Get all receipt IDs in original order (for consistent tensor columns)
        all_receipt_ids = df["receipt_id"].tolist()
        num_all_receipts = len(df)
        
        # Filter receipts early if receipt_ids is provided
        if receipt_ids is not None:
            receipt_ids_set = set(receipt_ids)
            # Create a mask for which receipts to include
            include_mask = df["receipt_id"].isin(receipt_ids_set)
            # But we still need to process all receipts for the full column set
            working_df = df
        else:
            include_mask = pd.Series([True] * len(df))
            working_df = df
        
        # Collect all unique products from the INCLUDED receipts only
        all_products = set()
        for idx, prod_list in enumerate(working_df["products"]):
            if include_mask.iloc[idx] if receipt_ids is not None else True:
                all_products.update(prod_list.split())
        # --- PATCH: Exclude products if requested ---
        if exclude_products is not None:
            exclude_products_set = set(str(p) for p in exclude_products)
            all_products = {p for p in all_products if p not in exclude_products_set}
        # ...existing code...
        all_products = sorted(list(all_products))
        product_to_idx = {p: i for i, p in enumerate(all_products)}
        idx_to_product = {i: p for p, i in product_to_idx.items()}
        num_products = len(all_products)
        # ...existing code...
        # Build sparse matrix - only include data for requested receipts
        rows = []
        cols = []
        data = []

        for receipt_idx, (receipt_id, prod_list) in enumerate(zip(working_df["receipt_id"], working_df["products"])):
            if include_mask.iloc[receipt_idx] if receipt_ids is not None else True:
                for p in prod_list.split():
                    if p in product_to_idx:  # Only add if not excluded
                        rows.append(product_to_idx[p])
                        cols.append(receipt_idx)
                        data.append(1)

        # Always create matrix with shape for ALL receipts
        features_sparse = csr_matrix((data, (rows, cols)), shape=(num_products, num_all_receipts))
        logger.log_operation("allocation", "CSR sparse matrix created", __file__, inspect.currentframe().f_lineno, size=features_sparse.data.nbytes + features_sparse.indptr.nbytes + features_sparse.indices.nbytes)
        
        # Convert to PyTorch sparse tensor - FIXED VERSION
        coo = features_sparse.tocoo()
        logger.log_operation("allocation", "COO sparse matrix created", __file__, inspect.currentframe().f_lineno, size=coo.data.nbytes + coo.row.nbytes + coo.col.nbytes)

        # Convert to single numpy array first to avoid the warning
        indices_np = np.array([coo.row, coo.col])
        logger.log_operation("allocation", "Numpy indices array created", __file__, inspect.currentframe().f_lineno, size=indices_np.nbytes)
        indices = torch.from_numpy(indices_np).long()
        logger.log_operation("allocation", "Torch indices tensor created", __file__, inspect.currentframe().f_lineno, size=indices.element_size() * indices.nelement())

        values = torch.from_numpy(coo.data).float()
        logger.log_operation("allocation", "Torch values tensor created", __file__, inspect.currentframe().f_lineno, size=values.element_size() * values.nelement())

        # Use the recommended sparse_coo_tensor constructor
        features_sparse_tensor = torch.sparse_coo_tensor(
            indices, 
            values, 
            torch.Size(coo.shape),
            dtype=torch.float32
        )
        logger.log_operation("allocation", "Torch sparse_coo_tensor created", __file__, inspect.currentframe().f_lineno, size=features_sparse_tensor._nnz() * features_sparse_tensor.element_size())
        
        print(f"Sparse feature matrix shape: {features_sparse_tensor.shape}")
        print(f"Total receipts in tensor: {num_all_receipts}")
        if receipt_ids is not None:
            num_included = include_mask.sum()
            print(f"Receipts with features: {num_included} out of {len(receipt_ids)} requested")
        
        return features_sparse_tensor, product_to_idx, idx_to_product, all_receipt_ids


    # -------------------
    # Encoder-Decoder MLP
    # -------------------
    class MLP_AutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, emb_dim):
            super().__init__()
            # Encoder MLP
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, emb_dim)
            )
            # Decoder MLP (separate)
            self.decoder = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

        def get_embedding(self, features):
            if features.is_sparse:
                logger.log_operation("transfer", "Converting sparse to dense for embedding", __file__, inspect.currentframe().f_lineno, size=features._nnz() * features.element_size())
                features = features.to_dense()
            return self.encoder(features)

        def forward(self, features):
            if features.is_sparse:
                logger.log_operation("transfer", "Converting sparse to dense for forward", __file__, inspect.currentframe().f_lineno, size=features._nnz() * features.element_size())
                features = features.to_dense()
            z = self.encoder(features)
            x_recon = self.decoder(z)
            return z, x_recon

    # -------------------
    # Hierarchical loss stub
    # -------------------
    def hierarchy_contrastive_loss(z, node_categories, temperature=0.2, level_weights=[0.4, 0.3, 0.2, 0.1], eps=1e-6):
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

        # Mask self-similarity using a dtype-safe negative infinity
        mask_eye = torch.eye(N, device=device).bool()
        neg_inf = torch.tensor(-float('inf'), device=device, dtype=sim.dtype)
        sim = sim.masked_fill(mask_eye, neg_inf)

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

    # -------------------
    # Training loop with batching
    # -------------------
    def train_model(
        model,
        features,               # [num_nodes, num_receipts], sparse
        node_categories,        # dict: {level_name: tensor_of_labels}
        epochs=300,
        lr=0.001,
        batch_size=128,
        eval_every=10,
        val_features=None,
        val_node_categories=None,
        lambda_hierarchy=0.3,
        device='cuda',
        recon_pos_weight=10.0
    ):
        print("Starting training...")
        logger.log_operation("transfer", "Moving model to device", __file__, inspect.currentframe().f_lineno)
        model = model.to(device)
        
        # Coalesce sparse tensors first
        print("Coalescing sparse tensors...")
        logger.log_operation("allocation", "Coalescing and moving features to CPU", __file__, inspect.currentframe().f_lineno, size=features._nnz() * features.element_size())
        features = features.coalesce().cpu()
        
        # Handle node_categories - keep on CPU
        node_categories_cpu = {}
        if node_categories is not None:
            for level_name, category_tensor in node_categories.items():
                if torch.is_tensor(category_tensor):
                    logger.log_operation("transfer", f"Moving node_categories[{level_name}] to CPU", __file__, inspect.currentframe().f_lineno, size=category_tensor.element_size() * category_tensor.nelement())
                    node_categories_cpu[level_name] = category_tensor.cpu()
                else:
                    node_categories_cpu[level_name] = torch.tensor(category_tensor, dtype=torch.long).cpu()
        
        # Handle validation data - keep on CPU
        val_node_categories_cpu = {}
        if val_features is not None:
            logger.log_operation("allocation", "Coalescing and moving val_features to CPU", __file__, inspect.currentframe().f_lineno, size=val_features._nnz() * val_features.element_size())
            val_features = val_features.coalesce().cpu()
            if val_node_categories is not None:
                for level_name, category_tensor in val_node_categories.items():
                    if torch.is_tensor(category_tensor):
                        logger.log_operation("transfer", f"Moving val_node_categories[{level_name}] to CPU", __file__, inspect.currentframe().f_lineno, size=category_tensor.element_size() * category_tensor.nelement())
                        val_node_categories_cpu[level_name] = category_tensor.cpu()
                    else:
                        val_node_categories_cpu[level_name] = torch.tensor(category_tensor, dtype=torch.long).cpu()

        # Use AdamW with 8-bit optimizer state to save memory
        # This can save up to 75% of optimizer memory!
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=1e-7)
            print("Using 8-bit AdamW optimizer (memory efficient)")
        except ImportError:
            # Fallback: Use AdamW with foreach=False (more memory efficient than Adam)
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=1e-7,
                foreach=False,  # Slightly slower but more memory efficient
                fused=False     # Disable fused implementation for lower memory
            )
            print("Using standard AdamW optimizer")
        
        scheduler = None
        if val_features is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-7
            )

        scaler = GradScaler()
        
        # Enable memory efficient settings
        torch.backends.cudnn.benchmark = False  # Disable for lower memory
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for efficiency
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        num_nodes = features.shape[0]

        def get_sparse_batch(sparse_tensor, batch_indices):
            """Extract a batch from sparse tensor using indices"""
            indices = sparse_tensor.indices()
            values = sparse_tensor.values()
            
            # Create a mask for rows in batch_indices
            batch_set = set(batch_indices.tolist())
            row_mask = torch.tensor([idx.item() in batch_set for idx in indices[0]], dtype=torch.bool)
            
            if not row_mask.any():
                # Empty batch
                return torch.zeros((len(batch_indices), sparse_tensor.shape[1]), 
                                dtype=sparse_tensor.dtype, device=device)
            
            # Filter indices and values
            batch_sparse_indices = indices[:, row_mask].clone()
            batch_values = values[row_mask]
            
            # Remap row indices to 0-based batch indices
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(batch_indices.tolist())}
            for i in range(batch_sparse_indices.shape[1]):
                batch_sparse_indices[0, i] = old_to_new[batch_sparse_indices[0, i].item()]
            
            # Create sparse tensor on CPU, convert to dense, pin and transfer
            logger.log_operation("allocation", "Creating batch_sparse_cpu", __file__, inspect.currentframe().f_lineno, size=batch_sparse_indices.element_size() * batch_sparse_indices.nelement() + batch_values.element_size() * batch_values.nelement())
            batch_sparse_cpu = torch.sparse_coo_tensor(
                batch_sparse_indices.cpu(),
                batch_values.cpu(),
                (len(batch_indices), sparse_tensor.shape[1]),
                dtype=sparse_tensor.dtype,
                device='cpu'
            )
            logger.log_operation("allocation", "Converting batch_sparse_cpu to dense", __file__, inspect.currentframe().f_lineno, size=batch_sparse_cpu._nnz() * batch_sparse_cpu.element_size())
            batch_dense = batch_sparse_cpu.to_dense().pin_memory()
            logger.log_operation("transfer", "Moving batch_dense to device", __file__, inspect.currentframe().f_lineno, size=batch_dense.element_size() * batch_dense.nelement())
            batch_dense = batch_dense.to(device, non_blocking=True)
            return batch_dense

        # Pre-allocate shuffle indices on CPU
        shuffle_indices = torch.arange(num_nodes, device='cpu')
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            # Shuffle indices for training (in-place on CPU)
            shuffle_indices = shuffle_indices[torch.randperm(num_nodes)]
            
            for start in range(0, num_nodes, batch_size):
                end = min(start + batch_size, num_nodes)
                batch_indices = shuffle_indices[start:end]
                
                # Get batch data from sparse tensor
                batch_x = get_sparse_batch(features, batch_indices)
                
                # Get batch categories for each level - transfer directly to device
                batch_cats = {}
                for level_name, cat_tensor in node_categories_cpu.items():
                    logger.log_operation("transfer", f"Moving batch_cats[{level_name}] to device", __file__, inspect.currentframe().f_lineno, size=cat_tensor[batch_indices].element_size() * cat_tensor[batch_indices].nelement())
                    batch_cats[level_name] = cat_tensor[batch_indices].to(device, non_blocking=True)
                
                logger.log_operation("deallocation", "Zeroing optimizer gradients", __file__, inspect.currentframe().f_lineno)
                optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()

                with autocast(device_type=device):
                    z_batch, x_recon_batch = model(batch_x)
                    #recon_loss = F.mse_loss(x_recon_batch, batch_x)
                    pos_w = torch.tensor(recon_pos_weight, device=x_recon_batch.device, dtype=torch.float32)
                    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction='mean')
                    recon_loss = bce_fn(x_recon_batch, batch_x)
                    # for monitoring compute recall on ones
                    with torch.no_grad():
                        probs = torch.sigmoid(x_recon_batch)
                        preds = (probs > 0.5).float()
                        total_ones = batch_x.sum()
                        if total_ones.item() > 0:
                            tp = (preds * batch_x).sum()
                            batch_recall_ones = (tp / (total_ones + 1e-12)).item()
                        else:
                            batch_recall_ones = 0.0
                    hier_loss = hierarchy_contrastive_loss(z_batch, batch_cats)
                    batch_loss = (1-lambda_hierarchy) * recon_loss + lambda_hierarchy * hier_loss

                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += batch_loss.item() * (end - start)
                num_batches += 1
                print (f"\rEpoch {epoch+1:03d} | Batch {num_batches} / {((num_nodes - 1) // batch_size) + 1} | Batch Loss: {batch_loss.item():.4f} | Recon Loss: {recon_loss.item():.4f} | Hier Loss: {hier_loss.item():.4f} | RecallOnOnes: {batch_recall_ones:.3f}", end='')
                # Delete batch tensors to free memory immediately
                logger.log_operation("deallocation", "Deleting batch tensors", __file__, inspect.currentframe().f_lineno)
                del batch_x, batch_cats, z_batch, x_recon_batch, batch_loss, recon_loss, hier_loss

            total_loss /= num_nodes
            history['train_loss'].append(total_loss)

            # Logging
            #if (epoch + 1) % 10 == 0:
            logger.log_operation("log", f"Epoch {epoch+1:03d} | Train Loss: {total_loss:.4f}", __file__, inspect.currentframe().f_lineno)
            print(f" | Train Loss: {total_loss:.4f}")

            # Validation
            if val_features is not None and (epoch + 1) % eval_every == 0:
                model.eval()
                val_total_loss = 0.0
                num_val_nodes = val_features.shape[0]
                
                # Create validation indices
                val_indices = torch.arange(num_val_nodes, device='cpu')
                
                with torch.no_grad():
                    for start in range(0, num_val_nodes, batch_size):
                        end = min(start + batch_size, num_val_nodes)
                        batch_val_indices = val_indices[start:end]
                        val_x = get_sparse_batch(val_features, batch_val_indices)
                        
                        # Get validation batch categories
                        val_batch_cats = {}
                        for level_name, cat_tensor in val_node_categories_cpu.items():
                            logger.log_operation("transfer", f"Moving val_batch_cats[{level_name}] to device", __file__, inspect.currentframe().f_lineno, size=cat_tensor[batch_val_indices].element_size() * cat_tensor[batch_val_indices].nelement())
                            val_batch_cats[level_name] = cat_tensor[batch_val_indices].to(device, non_blocking=True)

                        with autocast(device_type='cuda'):
                            z_val, x_val_recon = model(val_x)
                            #recon_loss = F.mse_loss(x_val_recon, val_x)
                            pos_w = torch.tensor(recon_pos_weight, device=x_recon_batch.device, dtype=torch.float32)
                            bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction='mean')
                            recon_loss = bce_fn(x_recon_batch, batch_x)
                            # for monitoring compute recall on ones
                            with torch.no_grad():
                                probs = torch.sigmoid(x_recon_batch)
                                preds = (probs > 0.5).float()
                                total_ones = batch_x.sum()
                                if total_ones.item() > 0:
                                    tp = (preds * batch_x).sum()
                                    batch_recall_ones = (tp / (total_ones + 1e-12)).item()
                                else:
                                    batch_recall_ones = 0.0
                            hier_loss = hierarchy_contrastive_loss(z_val, val_batch_cats)
                            val_batch_loss = recon_loss + lambda_hierarchy * hier_loss
                        
                        val_total_loss += val_batch_loss.item() * (end - start)
                        val_total_recall_ones += batch_recall_ones * (end - start)
                        
                        # Clean up validation batch
                        logger.log_operation("deallocation", "Deleting validation batch tensors", __file__, inspect.currentframe().f_lineno)
                        del val_x, val_batch_cats, z_val, x_val_recon, recon_loss, hier_loss, val_batch_loss

                val_total_loss /= num_val_nodes
                val_total_recall_ones /= num_val_nodes
                history['val_loss'].append(val_total_loss)
                history['val_recall_ones'].append(val_total_recall_ones)
                logger.log_operation("log", f"Validation Loss: {val_total_loss:.4f} | RecallOnOnes: {val_total_recall_ones:.3f}", __file__, inspect.currentframe().f_lineno)
                print(f"Validation Loss: {val_total_loss:.4f} | RecallOnOnes: {val_total_recall_ones:.3f}")

                if scheduler is not None:
                    scheduler.step(val_total_loss)

                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    logger.log_operation("allocation", "Saving best model state_dict", __file__, inspect.currentframe().f_lineno)
                    torch.save(model.state_dict(), "mlp_autoencoder_best.pth")
                    print(f"New best model saved with val loss: {best_val_loss:.4f}")

            # Memory management - more aggressive
            if epoch % 5 == 0:  # More frequent cache clearing
                logger.log_operation("deallocation", "Clearing CUDA cache", __file__, inspect.currentframe().f_lineno)
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        logger.log_operation("allocation", "Saving final model state_dict", __file__, inspect.currentframe().f_lineno)
        print(f"Training completed. Best val loss: {best_val_loss:.4f}")
        torch.save(model.state_dict(), "mlp_autoencoder_final.pth")
        return history

    with open("../data/train_receipt_ids.pkl", "rb") as f:
        train_receipt_ids = pickle.load(f)
    with open("../data/val_receipt_ids.pkl", "rb") as f:
        val_receipt_ids = pickle.load(f)

    train_features, train_product_to_idx, idx_to_product, all_receipt_ids = load_receipt_features("../data/grouped_products.csv", receipt_ids=train_receipt_ids)
    print('train features loaded')
    val_features, val_product_to_idx, idx_to_product, all_receipt_ids = load_receipt_features("../data/grouped_products.csv", receipt_ids=val_receipt_ids, exclude_products=list(train_product_to_idx.keys()))
    print('val features loaded - length:', val_features.shape)
    with open("../data/metadata_labels.pkl", "rb") as f:
        metadata = pickle.load(f)

    train_node_labels = simple_align_labels(metadata, train_product_to_idx)
    val_node_labels = simple_align_labels(metadata, val_product_to_idx)

    def convert_labels_to_tensor(node_labels, device):
        node_categories = {}
        levels_names = ["descr_liv1", "descr_liv2", "descr_liv3", "descr_liv4"]
        for level in levels_names:
            if level in node_labels.columns:
                # Get unique labels and create mapping
                unique_labels = node_labels[level].unique()
                label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
                
                # Convert labels to tensor using the mapping
                mapped_labels = node_labels[level].map(label_to_id)
                node_categories[level] = torch.tensor(
                    mapped_labels.values, 
                    dtype=torch.long,
                    device="cpu"
                )
        return node_categories

    train_node_labels = convert_labels_to_tensor(train_node_labels, device='cuda')
    val_node_labels = convert_labels_to_tensor(val_node_labels, device='cuda')

    num_nodes, num_receipts = train_features.shape

    model = MLP_AutoEncoder(
        input_dim=num_receipts,
        hidden_dim=128,
        emb_dim=64
    )

    history = train_model(
        model,
        train_features,
        train_node_labels,
        epochs=50,
        lr=0.001,
        batch_size=256,
        val_features=val_features,
        val_node_categories=val_node_labels,
        lambda_hierarchy=0.8,
        device='cuda'
    )


