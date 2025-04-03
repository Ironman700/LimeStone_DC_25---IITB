import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle
import json

##############################
# Data Loader and Dataset
##############################
def data_loader(path, table_idx, player_or_dealer):
    # Utility for loading train.csv; returns an array with spy and card values.
    data = pd.read_csv(path, header=[0, 1, 2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

def create_dataset(series, window_size=5):
    """
    Given a 1D numpy array 'series', create input-output pairs.
    Each input is a window of consecutive 'window_size' spy values and the output is the next spy value.
    """
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

##############################
# LSTM Model Definition
##############################
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        """
        A simple LSTM-based regressor.
        - input_size: number of features per time step (1 for a univariate series).
        - hidden_size: dimensionality of LSTM hidden state.
        - num_layers: number of stacked LSTM layers.
        - output_size: dimension of the output (1 for scalar prediction).
        """
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

##############################
# Training and Evaluation
##############################
def train_model(model, X, y, learning_rate, num_epochs):
    """
    Train the given LSTM model using Mean Squared Error loss and the Adam optimizer.
    Continuously prints the training loss at each epoch.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
    model.eval()
    return model

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        loss = nn.MSELoss()(predictions, y)
    return loss.item()

def tune_model(X_train, y_train, X_val, y_val, hyperparams_grid):
    best_loss = float('inf')
    best_params = None
    best_model_state = None    
    # Grid search over hyperparameters.
    for hidden_size in hyperparams_grid['hidden_size']:
        for num_layers in hyperparams_grid['num_layers']:
            for lr in hyperparams_grid['lr']:
                for epochs in hyperparams_grid['epochs']:
                    model = LSTMRegressor(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
                    model = train_model(model, X_train, y_train, learning_rate=lr, num_epochs=epochs)
                    val_loss = evaluate_model(model, X_val, y_val)
                    print(f"Params: hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}, epochs={epochs} -> Val MSE: {val_loss:.4f}")
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_params = {'hidden_size': hidden_size, 'num_layers': num_layers, 'lr': lr, 'epochs': epochs}
                        best_model_state = model.state_dict()
    return best_params, best_model_state, best_loss

def save_model(checkpoint, filepath):
    torch.save(checkpoint, filepath)

def load_model(filepath):
    # Load the checkpoint with weights_only=True to limit risk.
    checkpoint = torch.load(filepath, weights_only=True)
    if 'hyperparams' not in checkpoint:
        raise KeyError(
            "Checkpoint file does not contain 'hyperparams'. "
            "This is likely from an older training run. Please delete the old checkpoint file "
            f"({filepath}) and retrain the model."
        )
    hyperparams = checkpoint['hyperparams']
    model = LSTMRegressor(input_size=1, hidden_size=hyperparams['hidden_size'],
                          num_layers=hyperparams['num_layers'], output_size=1)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, hyperparams

##############################
# MyPlayer Class
##############################
class MyPlayer:
    def __init__(self, table_index):
        """
        For a given table index (0 to 4), load training data from "train.csv" for both the player and dealer spy series.
        For each series, create training samples using a sliding window of 5 spy values to predict the next spy value.
        Then, hyperparameter tune and train an LSTM model on each series.
        StandardScaler normalization is applied, and the trained models and scalers are saved to disk.
        """
        self.table_index = table_index
        self.train_path = "train.csv"
        self.window_size = 5
        
        # Define file paths.
        self.player_model_path = f"model_table_{table_index}_player.pt"
        self.dealer_model_path = f"model_table_{table_index}_dealer.pt"
        self.player_scaler_path = f"scaler_table_{table_index}_player.pkl"
        self.dealer_scaler_path = f"scaler_table_{table_index}_dealer.pkl"
        
        # Load training data.
        player_data = data_loader(self.train_path, table_index, "player")
        dealer_data = data_loader(self.train_path, table_index, "dealer")
        player_spy = player_data[:, 0]
        dealer_spy = dealer_data[:, 0]
        
        # Fit scalers.
        self.scaler_player = StandardScaler()
        self.scaler_dealer = StandardScaler()
        player_spy_scaled = self.scaler_player.fit_transform(player_spy.reshape(-1, 1)).flatten()
        dealer_spy_scaled = self.scaler_dealer.fit_transform(dealer_spy.reshape(-1, 1)).flatten()
        
        # Save or load scalers.
        if not os.path.exists(self.player_scaler_path):
            with open(self.player_scaler_path, 'wb') as f:
                pickle.dump(self.scaler_player, f)
        else:
            with open(self.player_scaler_path, 'rb') as f:
                self.scaler_player = pickle.load(f)
        if not os.path.exists(self.dealer_scaler_path):
            with open(self.dealer_scaler_path, 'wb') as f:
                pickle.dump(self.scaler_dealer, f)
        else:
            with open(self.dealer_scaler_path, 'rb') as f:
                self.scaler_dealer = pickle.load(f)
        
        # Create sliding window datasets.
        X_player_np, y_player_np = create_dataset(player_spy_scaled, self.window_size)
        X_dealer_np, y_dealer_np = create_dataset(dealer_spy_scaled, self.window_size)
        self.X_player = torch.tensor(X_player_np, dtype=torch.float32).unsqueeze(-1)
        self.y_player = torch.tensor(y_player_np, dtype=torch.float32).unsqueeze(-1)
        self.X_dealer = torch.tensor(X_dealer_np, dtype=torch.float32).unsqueeze(-1)
        self.y_dealer = torch.tensor(y_dealer_np, dtype=torch.float32).unsqueeze(-1)
        
        # Hyperparameter grid.
        hyperparams_grid = {
            'hidden_size': [16, 32],
            'num_layers': [1, 2],
            'lr': [0.001, 0.01],
            'epochs': [100]
        }
        
        # Train or load player model.
        if os.path.exists(self.player_model_path):
            self.model_player, _ = load_model(self.player_model_path)
        else:
            split_idx = int(0.8 * self.X_player.shape[0])
            X_train = self.X_player[:split_idx]
            y_train = self.y_player[:split_idx]
            X_val = self.X_player[split_idx:]
            y_val = self.y_player[split_idx:]
            best_params, best_state, best_loss = tune_model(X_train, y_train, X_val, y_val, hyperparams_grid)
            print(f"Best hyperparameters for table {table_index} player: {best_params} with val MSE {best_loss:.4f}")
            self.model_player = LSTMRegressor(input_size=1, hidden_size=best_params['hidden_size'],
                                              num_layers=best_params['num_layers'], output_size=1)
            self.model_player = train_model(self.model_player, self.X_player, self.y_player,
                                            learning_rate=best_params['lr'], num_epochs=best_params['epochs'])
            checkpoint = {'state_dict': self.model_player.state_dict(), 'hyperparams': best_params}
            save_model(checkpoint, self.player_model_path)
        
        # Train or load dealer model.
        if os.path.exists(self.dealer_model_path):
            self.model_dealer, _ = load_model(self.dealer_model_path)
        else:
            split_idx = int(0.8 * self.X_dealer.shape[0])
            X_train = self.X_dealer[:split_idx]
            y_train = self.y_dealer[:split_idx]
            X_val = self.X_dealer[split_idx:]
            y_val = self.y_dealer[split_idx:]
            best_params, best_state, best_loss = tune_model(X_train, y_train, X_val, y_val, hyperparams_grid)
            print(f"Best hyperparameters for table {table_index} dealer: {best_params} with val MSE {best_loss:.4f}")
            self.model_dealer = LSTMRegressor(input_size=1, hidden_size=best_params['hidden_size'],
                                              num_layers=best_params['num_layers'], output_size=1)
            self.model_dealer = train_model(self.model_dealer, self.X_dealer, self.y_dealer,
                                            learning_rate=best_params['lr'], num_epochs=best_params['epochs'])
            checkpoint = {'state_dict': self.model_dealer.state_dict(), 'hyperparams': best_params}
            save_model(checkpoint, self.dealer_model_path)
        
    def get_card_value_from_spy_value(self, value):
        # Placeholder mapping: round the spy value.
        return int(round(value))
        
    def get_player_spy_prediction(self, hist):
        hist = np.array(hist).reshape(-1, 1)
        hist_scaled = self.scaler_player.transform(hist).flatten()
        hist_tensor = torch.tensor(hist_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            prediction_scaled = self.model_player(hist_tensor).item()
        prediction = self.scaler_player.inverse_transform([[prediction_scaled]])[0,0]
        return prediction

    def get_dealer_spy_prediction(self, hist):
        hist = np.array(hist).reshape(-1, 1)
        hist_scaled = self.scaler_dealer.transform(hist).flatten()
        hist_tensor = torch.tensor(hist_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            prediction_scaled = self.model_dealer(hist_tensor).item()
        prediction = self.scaler_dealer.inverse_transform([[prediction_scaled]])[0,0]
        return prediction

    def get_player_action(self,
                          curr_spy_history_player, 
                          curr_spy_history_dealer, 
                          curr_card_history_player, 
                          curr_card_history_dealer, 
                          curr_player_total, 
                          curr_dealer_total, 
                          turn,
                          game_index):
        next_spy_player = self.get_player_spy_prediction(curr_spy_history_player)
        next_spy_dealer = self.get_dealer_spy_prediction(curr_spy_history_dealer)
        next_pred_card_player = self.get_card_value_from_spy_value(next_spy_player)
        next_pred_card_dealer = self.get_card_value_from_spy_value(next_spy_dealer)
        
        if turn == 'player':
            curr_player_total += next_pred_card_player
            if curr_player_total > 20:
                return 'stand'
            return 'hit'
        else:
            curr_dealer_total += next_pred_card_dealer
            if curr_dealer_total > 21:
                return 'continue'
            else :
                if curr_dealer_total > curr_player_total:
                    return 'surrender'
                else:    
                    return 'continue'