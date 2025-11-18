# file: geonera-ai/selection/phase2_clustering.py

import logging
import pandas as pd
import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .common import _split_data_components


def run_phase2_clustering(
        df: pd.DataFrame,
        correlation_threshold: float = 0.95
) -> pd.DataFrame | None:
    logging.info(f"Phase 2 (Spearman + Clustering) Input Shape: {df.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda":
        logging.info("Phase 2: Found GPU. Correlation matrix will use CUDA.")
    else:
        logging.warning("Phase 2: GPU not found. Correlation will run on CPU (this may be slow).")
    split_result = _split_data_components(df)
    if split_result is None:
        logging.error("Phase 2: Failed to split data components. Aborting.")
        return None
    id_cols, target_cols, feature_cols, protected_cols = split_result
    X_df = df[feature_cols]
    if X_df.empty:
        logging.warning("Phase 2: No features found to cluster. Skipping.")
        return df
    torch.cuda.empty_cache()
    try:
        logging.info(f"Phase 2: Moving {X_df.shape[1]} features to {device} for correlation matrix...")
        X_tensor = torch.tensor(X_df.values, dtype=torch.float32).to(device)
        X_ranks = torch.argsort(torch.argsort(X_tensor, dim=0), dim=0).float()
        logging.info("Phase 2: Calculating Spearman correlation matrix on GPU...")
        X_ranks_mean = X_ranks.mean(dim=0, keepdim=True)
        X_ranks_std = X_ranks.std(dim=0, keepdim=True)
        X_ranks_std[X_ranks_std == 0] = 1.0
        X_ranks_norm = (X_ranks - X_ranks_mean) / X_ranks_std
        corr_matrix_gpu = torch.corrcoef(X_ranks_norm.T)
        logging.info("Phase 2: Moving correlation matrix to CPU for clustering...")
        corr_matrix_cpu = corr_matrix_gpu.cpu().numpy()
        del X_tensor, X_ranks, X_ranks_mean, X_ranks_std, X_ranks_norm, corr_matrix_gpu
        torch.cuda.empty_cache()
        distance_matrix = 1 - np.abs(corr_matrix_cpu)
        distance_matrix[np.isnan(distance_matrix)] = 1
        distance_matrix[distance_matrix < 0] = 0
        np.fill_diagonal(distance_matrix, 0)
        condensed_dist = squareform(distance_matrix, checks=False)
        logging.info("Phase 2: Running Hierarchical Agglomerative Clustering on CPU...")
        Z = linkage(condensed_dist, method='average')
        cluster_labels = fcluster(Z, t=(1 - correlation_threshold), criterion='distance')
        logging.info("Phase 2: Selecting representatives from clusters...")
        n_clusters = cluster_labels.max()
        kept_features = []
        feature_names = X_df.columns.tolist()
        for i in range(1, n_clusters + 1):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) > 0:
                representative_index = cluster_indices[0]
                kept_features.append(feature_names[representative_index])
        dropped_count = len(feature_names) - len(kept_features)
        logging.info(f"Phase 2: Kept {len(kept_features)} cluster representatives. Dropped {dropped_count} redundant features.")
        X_selected_df = X_df[kept_features]
        final_df = pd.concat([df[id_cols], df[target_cols], df[protected_cols], X_selected_df], axis=1)
        logging.info(f"Phase 2 (Clustering): Feature reduction complete. Final shape: {final_df.shape}.")
        return final_df

    except Exception as e:
        logging.exception(f"Phase 2: An unrecoverable error occurred during clustering: {e}")
        return None
    finally:
        torch.cuda.empty_cache()