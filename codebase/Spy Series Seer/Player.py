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

def data_loader(path, table_idx, player_or_dealer):
    # Utility for loading train.csv; returns an array with spy and card values.
    data = pd.read_csv(path, header=[0,1,2])
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
    X = np.array(X)  # shape: (num_samples, window_size)
    y = np.array(y)  # shape: (num_samples,)
    return X, y

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
        
        # LSTM layer with batch_first=True so that input has shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        # Initialize hidden and cell states with zeros.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Use the last output of the sequence.
        out = self.fc(out[:, -1, :])
        return out

class MyPlayer:
    def __init__(self, table_index):
        """
        For a given table index (0 to 4), load training data from "train.csv" for both the player and dealer spy series.
        For each series, create training samples using a sliding window of 5 spy values to predict the next spy value.
        Then, train an LSTM model on each series.
        
        We also apply StandardScaler normalization because the player spy series in particular shows high variance.
        After training, predictions will be inverse-transformed to get the actual scale.
        """
        self.table_index = table_index
        self.train_path = "train.csv"   # Assumes the training file is named "train.csv"
        self.window_size = 5
        
        # Load training data for player and dealer streams.
        player_data = data_loader(self.train_path, table_index, "player")
        dealer_data = data_loader(self.train_path, table_index, "dealer")
        
        # Extract spy series (first column)
        player_spy = player_data[:, 0]
        dealer_spy = dealer_data[:, 0]
        
        # Fit separate scalers for player and dealer
        self.scaler_player = StandardScaler()
        self.scaler_dealer = StandardScaler()
        player_spy_scaled = self.scaler_player.fit_transform(player_spy.reshape(-1, 1)).flatten()
        dealer_spy_scaled = self.scaler_dealer.fit_transform(dealer_spy.reshape(-1, 1)).flatten()
        
        # Save the original series for potential further use
        self.player_spy_series = player_spy
        self.dealer_spy_series = dealer_spy
        
        # Create sliding window datasets on the scaled data.
        X_player, y_player = create_dataset(player_spy_scaled, self.window_size)
        X_dealer, y_dealer = create_dataset(dealer_spy_scaled, self.window_size)
        
        # Convert to PyTorch tensors and add feature dimension (input_size=1).
        self.X_player = torch.tensor(X_player, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, window_size, 1)
        self.y_player = torch.tensor(y_player, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, 1)
        self.X_dealer = torch.tensor(X_dealer, dtype=torch.float32).unsqueeze(-1)
        self.y_dealer = torch.tensor(y_dealer, dtype=torch.float32).unsqueeze(-1)
        
        # Initialize LSTM models for player and dealer.
        self.model_player = LSTMRegressor(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        self.model_dealer = LSTMRegressor(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        
        # Train the models.
        self.train_model(self.model_player, self.X_player, self.y_player, model_name="Player")
        self.train_model(self.model_dealer, self.X_dealer, self.y_dealer, model_name="Dealer")
        
    def train_model(self, model, X, y, model_name=""):
        """
        Train the given LSTM model on dataset (X, y) using Mean Squared Error loss and the Adam optimizer.
        """
        model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        num_epochs = 200
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 50 == 0:
                print(f"{model_name} Model Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        model.eval()
        
    def get_card_value_from_spy_value(self, value):
        """
        Given a spy value (float), return the corresponding card value.
        For this part we use a placeholder implementation (rounding to the nearest integer).
        In your final submission you may replace this with a more accurate deterministic function.
        
        Output:
            A scalar integer prediction for the card value.
        """
        return int(round(value))
        
    def get_player_spy_prediction(self, hist):
        """
        Given a history (1D numpy array of length 5) of player spy values, use the trained LSTM model to predict
        the next spy value.
        
        The history is first normalized using the player scaler, then fed to the model.
        The model's output is inverse-transformed to get the final prediction in the original scale.
        
        Output:
            A scalar float representing the predicted spy value.
        """
        # Normalize the history using the player's scaler.
        hist = np.array(hist).reshape(-1, 1)
        hist_scaled = self.scaler_player.transform(hist).flatten()
        
        # Convert to tensor of shape (1, window_size, 1)
        hist_tensor = torch.tensor(hist_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            prediction_scaled = self.model_player(hist_tensor).item()
        # Inverse transform to get the prediction in original scale.
        prediction = self.scaler_player.inverse_transform([[prediction_scaled]])[0,0]
        return prediction

    def get_dealer_spy_prediction(self, hist):
        """
        Given a history (1D numpy array of length 5) of dealer spy values, use the trained LSTM model to predict
        the next spy value.
        
        The history is normalized using the dealer scaler, then fed to the model.
        The output is inverse-transformed to get the prediction in the original scale.
        
        Output:
            A scalar float representing the predicted spy value.
        """
        hist = np.array(hist).reshape(-1, 1)
        hist_scaled = self.scaler_dealer.transform(hist).flatten()
        hist_tensor = torch.tensor(hist_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            prediction_scaled = self.model_dealer(hist_tensor).item()
        prediction = self.scaler_dealer.inverse_transform([[prediction_scaled]])[0,0]
        return prediction
