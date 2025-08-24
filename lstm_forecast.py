import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def make_sequences(df, feature_cols, target_col, window_size, pred_horizon=1):
    X, y, info = [], [], []
    arr = df[feature_cols].values
    arr_y = df[target_col].values
    idxs = df.index
    for i in range(len(df) - window_size - pred_horizon + 1):
        X.append(arr[i:i+window_size, :])
        y.append(arr_y[i+window_size+pred_horizon-1])
        info.append(idxs[i+window_size+pred_horizon-1])  # Mapping prediction to row
    return np.array(X), np.array(y), info

class DengueDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

def minmax_inverse(arr, min_, max_):
    return arr * (max_ - min_) + min_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.results_dir, exist_ok=True)

    CLIMATE_COLS = [
        'temp_min', 'temp_med', 'temp_max',
        'precip_min', 'precip_med', 'precip_max',
        'pressure_min', 'pressure_med', 'pressure_max',
        'rel_humid_min', 'rel_humid_med', 'rel_humid_max',
        'thermal_range', 'rainy_days'
    ]
    TARGET_VAR = 'casos'
    data = pd.read_csv(args.data_file)
    states = sorted(data['uf'].unique())
    tasks = [1, 2, 3]

    for task_num in tasks:
        train_col = f'train_{task_num}'
        target_col = f'target_{task_num}'

        print(f"\n=== Task {task_num} ===")
        for state in states:
            print(f"State {state}...")

            df_state = data[data['uf'] == state].sort_values(['year', 'epiweek']).reset_index(drop=True)
            if not (df_state[train_col].any() and df_state[target_col].any()):
                print(f"  Skip {state}: no train/target mask for task {task_num}")
                continue

            # Normalize features (fit on training period only to avoid data leakage)
            scaler = {}
            for col in CLIMATE_COLS + [TARGET_VAR]:
                train_mask = df_state[train_col]
                min_, max_ = df_state.loc[train_mask, col].min(), df_state.loc[train_mask, col].max()
                df_state[col] = (df_state[col] - min_) / (max_ - min_ + 1e-8)
                scaler[col] = (min_, max_)

            # 1. Train set
            df_train = df_state[df_state[train_col]].reset_index(drop=True)
            X_train, y_train, _ = make_sequences(df_train, [TARGET_VAR]+CLIMATE_COLS, TARGET_VAR, args.window_size)
            if len(X_train) == 0:
                print(f"  Not enough train samples in {state} for task {task_num}, skipping.")
                continue
            train_dataset = DengueDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            # 2. Target set (for prediction)
            df_target = df_state[df_state[target_col]].reset_index(drop=True)
            pred_idxs = df_state[df_state[target_col]].index
            preds = []
            for idx in pred_idxs:
                start_idx = idx - args.window_size
                if start_idx < 0:
                    preds.append(np.nan)
                    continue
                x_seq = df_state.loc[start_idx:idx-1, [TARGET_VAR]+CLIMATE_COLS].values
                x_seq = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                preds.append(x_seq)

            # ---- Define & Train Model ----
            input_dim = 1 + len(CLIMATE_COLS)
            model = SimpleLSTM(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.MSELoss()

            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0.0
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(Xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(Xb)
                if (epoch+1) % 10 == 0:
                    print(f"  Epoch {epoch+1:02d} Loss: {epoch_loss/len(train_loader.dataset):.4f}")

            # ---- Predict for target weeks ----
            model.eval()
            y_pred = []
            with torch.no_grad():
                for x_seq in preds:
                    if isinstance(x_seq, float) and np.isnan(x_seq):
                        y_pred.append(np.nan)
                        continue
                    out = model(x_seq)
                    y_pred.append(out.cpu().numpy().item())

            # Inverse scaling for predictions
            min_, max_ = scaler[TARGET_VAR]
            y_pred_unscaled = [minmax_inverse(v, min_, max_) if not np.isnan(v) else np.nan for v in y_pred]

            # Output DataFrame
            df_out = df_state.loc[pred_idxs, ['uf', 'epiweek', 'year', TARGET_VAR]].copy()
            df_out['predicted_casos'] = y_pred_unscaled
            df_out.reset_index(drop=True, inplace=True)
            # Save
            fname = f"{args.results_dir}/pred_task{task_num}_{state}.csv"
            df_out.to_csv(fname, index=False)
            print(f"  Saved predictions to {fname}")

if __name__ == "__main__":
    main()
