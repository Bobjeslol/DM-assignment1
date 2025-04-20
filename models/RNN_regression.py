import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import optuna
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import os
from sklearn.utils.class_weight import compute_class_weight
import optuna.visualization.matplotlib as optuna_vis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mood_sequences(csv_path, sequence_length=7):
    #Returns sequences and their corresponding targets.
    df = pd.read_csv(csv_path)
    
    grouped = df.groupby("sequence_id")
    sequences = []
    targets = []
    
    for _, group in grouped:
        group = group.sort_values("timestep")
        if len(group) != sequence_length:
            print(f"skipping sequence {group['sequence_id'].values[0]}")
            continue

        sequence = group[["var_id", "value", "delta_t"]].values
        target = group["target"].values[0]
        sequences.append(torch.tensor(sequence, dtype=torch.float32))
        targets.append(torch.tensor(target, dtype=torch.float32).unsqueeze(0))
    return sequences, targets


def create_dataloaders(train_csv, test_csv, batch_size=64, val_split=0.2):
    full_train_seqs, full_train_targets = load_mood_sequences(train_csv)
    test_seqs, test_targets = load_mood_sequences(test_csv)

    X_train, X_val, y_train, y_val = train_test_split(
        full_train_seqs, full_train_targets, test_size=val_split, random_state=42)

    train_dataset = TensorDataset(torch.stack(X_train), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.stack(X_val), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.stack(test_seqs), torch.tensor(test_targets, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader



def create_mood_model(num_var_ids, embedding_dim, lstm_hidden_size, lstm_layers, dropout):
    embedding = nn.Embedding(num_embeddings=num_var_ids, embedding_dim=embedding_dim)
    input_size = embedding_dim + 2
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=lstm_hidden_size,
        dropout=dropout if lstm_layers > 1 else 0,
        num_layers=lstm_layers,
        batch_first=True
    )
    fc = nn.Sequential(
        nn.Linear(lstm_hidden_size, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return embedding, lstm, fc


def predict_mood(x, embedding, lstm, fc):
    var_id = x[:, :, 0].long()
    value = x[:, :, 1].unsqueeze(-1)
    delta_t = x[:, :, 2].unsqueeze(-1)
    var_embed = embedding(var_id)
    lstm_input = torch.cat([var_embed, value, delta_t], dim=2)
    output, (h_n, _) = lstm(lstm_input)
    final_hidden = h_n[-1]
    return fc(final_hidden)


def evaluate_model(embedding, lstm, fc, dataloader, device="cpu", plot=True):
    embedding.eval(), lstm.eval(), fc.eval()
    total_mse = 0
    total_mae = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = predict_mood(batch_x, embedding, lstm, fc).squeeze()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.squeeze().cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    r2 = 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
    
    print(f"\nMean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    if plot:
        #predictions vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predictions vs Actual Values")
        plt.tight_layout()
        plt.show()
    return mse, mae, r2


def train_model(embedding, lstm, fc, train_loader, val_loader, epochs, lr, device="cpu", model_path=None, class_weights=None):
    model_params = list(embedding.parameters()) + list(lstm.parameters()) + list(fc.parameters())
    optimizer = optim.Adam(model_params, lr=lr)
    criterion = nn.MSELoss() 
    embedding.to(device), lstm.to(device), fc.to(device)

    best_mse = float('inf')
    patience, patience_counter = 5, 0

    for epoch in range(1, epochs + 1):
        embedding.train(), lstm.train(), fc.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = predict_mood(batch_x, embedding, lstm, fc).squeeze()
            loss = criterion(predictions, batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch}/{epochs} — Train loss: {total_loss / len(train_loader):.4f}")

        logger.info("Validation Evaluation:")
        val_mse, _, _ = evaluate_model(embedding, lstm, fc, val_loader, device, plot=False)

        if val_mse < best_mse:
            best_mse = val_mse
            patience_counter = 0
            if model_path:
                torch.save({
                    "embedding": embedding.state_dict(),
                    "lstm": lstm.state_dict(),
                    "fc": fc.state_dict()
                }, model_path)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break


def objective(trial):
    #Sample hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [4, 8, 16])
    lstm_hidden_size = trial.suggest_int("lstm_hidden_size", 32, 128, step=32)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Load data
    train_loader, val_loader, _ = create_dataloaders("train_RNN.csv", "test_RNN.csv", batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Create model
    embedding, lstm, fc = create_mood_model(
        num_var_ids=19,
        embedding_dim=embedding_dim,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout
    )

    #Training setup
    model_params = list(embedding.parameters()) + list(lstm.parameters()) + list(fc.parameters())
    optimizer = optim.Adam(model_params, lr=learning_rate)
    criterion = nn.MSELoss() 

    embedding.to(device), lstm.to(device), fc.to(device)

    #training for optuna trials
    for epoch in range(10):
        embedding.train(), lstm.train(), fc.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = predict_mood(batch_x, embedding, lstm, fc).squeeze()
            loss = criterion(predictions, batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #evaluate on validation
    val_mse, _, _ = evaluate_model(embedding, lstm, fc, val_loader, device, plot=False)

    #save best model
    if trial.number == 0 or val_mse < trial.study.best_value:
        torch.save({
            "embedding": embedding.state_dict(),
            "lstm": lstm.state_dict(),
            "fc": fc.state_dict()
        }, "best_model_optuna_regression.pt")

    return val_mse


def run_optuna():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)

    print("Best trial:")
    print(f"MSE: {study.best_value:.4f}")
    print("Params:", study.best_params)

    with open("optuna_results_regression.txt", "a") as f:
        f.write(f"MSE: {study.best_value:.4f}\nParams: {study.best_params}\n")

    #trial performance
    fig = optuna_vis.plot_optimization_history(study)
    plt.title("Optuna Optimization History: RNN Classification") 
    plt.tight_layout()
    plt.show()


def main():
    #gets best parameters from optuna_results.txt -> dictionary
    best_params = None
    best_mse = 999
    with open("optuna_results_regression.txt", "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
            mse = float(lines[i].split(": ")[1])
            if mse < best_mse:
                best_mse = mse
                params_str = lines[i + 1].split("Params: ")[1]
                best_params = eval(params_str)

    train_loader, val_loader, test_loader = create_dataloaders(
    "train_RNN.csv", 
    "test_RNN.csv", 
    batch_size=best_params['batch_size']
    )
    

    embedding, lstm, fc = create_mood_model(
        num_var_ids=19,
        embedding_dim=best_params['embedding_dim'],
        lstm_hidden_size=best_params['lstm_hidden_size'],
        lstm_layers=best_params['lstm_layers'],
        dropout=best_params['dropout']
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(
        embedding, lstm, fc,
        train_loader, val_loader,
        epochs=500,
        lr=best_params['lr'],
        device=device,
        model_path="final_model_regression.pt"
    )
    
    #model after training
    best_model = torch.load("final_model_regression.pt")
    embedding.load_state_dict(best_model['embedding'])
    lstm.load_state_dict(best_model['lstm'])
    fc.load_state_dict(best_model['fc'])
    
    print("\nFinal Evaluation on test set:")
    evaluate_model(embedding, lstm, fc, test_loader, device)


if __name__ == "__main__":
    run_optuna()
    #main()
