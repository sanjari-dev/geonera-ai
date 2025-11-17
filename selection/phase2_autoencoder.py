# file: geonera-ai/selection/phase2_autoencoder.py

import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .common import _split_data_components
from models import Autoencoder


def run_phase2_selection_autoencoder(
        df: pd.DataFrame,
        bottleneck_size: int = 100,
        num_epochs: int = 20,
        batch_size: int = 512
) -> pd.DataFrame | None:
    logging.info(f"Phase 2 (Autoencoder) Input Shape: {df.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda":
        logging.info("Phase 2: Found GPU. Autoencoder training will use CUDA.")
    else:
        logging.warning("Phase 2: GPU not found. Autoencoder will run on CPU (this may be slow).")

    split_result = _split_data_components(df)
    if split_result is None:
        logging.error("Phase 2: Failed to split data components. Aborting.")
        return None
    id_cols, target_cols, feature_cols, protected_cols = split_result

    X_df = df[feature_cols]
    y_df = df[target_cols]

    logging.info(f"Phase 2: Found {len(feature_cols)} features to compress.")
    if X_df.empty:
        logging.error("Phase 2: No features found to compress after splitting. Aborting.")
        return df  # Return original df, no features to compress

    logging.info("Phase 2: Applying StandardScaler to features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(feature_cols)
    model = Autoencoder(input_dim, bottleneck_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    logging.info(f"Phase 2: Starting Autoencoder training for {num_epochs} epochs...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for (inputs, targets) in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            logging.info(f"Phase 2: Epoch [{epoch + 1}/{num_epochs}], Reconstruction Loss: {avg_loss:.6f}")

    logging.info(f"Phase 2: Training complete. Extracting {bottleneck_size} meta-features...")
    model.eval()
    with torch.no_grad():
        X_full_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        encoded_features_tensor = model.encoder(X_full_tensor)

    encoded_features_np = encoded_features_tensor.cpu().numpy()

    meta_feature_names = [f"AE_meta_{i}" for i in range(bottleneck_size)]
    X_selected_df = pd.DataFrame(encoded_features_np, columns=meta_feature_names, index=X_df.index)

    final_df = pd.concat([df[id_cols], y_df, df[protected_cols], X_selected_df], axis=1)

    logging.info(f"Phase 2 (Autoencoder): Feature compression complete. Final shape: {final_df.shape}.")

    return final_df