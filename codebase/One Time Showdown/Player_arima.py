import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Module-level cache for trained models/scalers by table index.
_trained_models = {}  # key: table_index, value: (scaler_player, scaler_dealer, player_arima_params, dealer_arima_params)

def spy_to_val_train_model(csv_path: str) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier on the provided CSV data.
    
    The CSV file is assumed to have a three-row header:
      - Row 1: table identifiers (e.g., table_0, table_1, ...)
      - Row 2: stream type (player or dealer)
      - Row 3: attribute ('spy' or 'card')
      
    This function collects all (spy, card) pairs from all tables/streams and trains
    a decision tree classifier to perfectly capture the deterministic mapping.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        
    Returns:
        DecisionTreeClassifier: A trained classifier.
    """
    # Read CSV file with a multi-index header (3 rows)
    data = pd.read_csv(csv_path, header=[0, 1, 2])
    X = []
    y = []
    
    # Iterate over columns and look for those corresponding to the spy values.
    for col in data.columns:
        if col[2] == 'spy':
            card_col = (col[0], col[1], 'card')
            if card_col in data.columns:
                spy_vals = data[col].values
                card_vals = data[card_col].values
                X.extend(spy_vals.tolist())
                y.extend(card_vals.tolist())
    
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    # Train a Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model

SPY_To_Val_MODEL = spy_to_val_train_model("train.csv")

def data_loader(path, table_idx, player_or_dealer):
    # Utility for loading train.csv, example use in the notebook.
    # player_or_dealer can either be "player" or "dealer"
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

def find_best_arima_params(series, max_p=3, max_d=2, max_q=3):
    """
    Grid search for the best ARIMA parameters based on AIC.
    Returns the best (p,d,q) tuple.
    """
    best_aic = float('inf')
    best_params = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                # Skip (0,0,0) as it's a trivial model
                if p == 0 and d == 0 and q == 0:
                    continue
                    
                try:
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                except:
                    continue
    
    # If no valid model is found, use a simple AR(1) model
    if best_params is None:
        best_params = (1, 0, 0)
        
    return best_params

class MyPlayer:
    def __init__(self, table_index):
        """
        For a given table index (0 to 4), load training data from "train.csv" for both the player and dealer spy series.
        Find the best ARIMA parameters for each series and store them.
        
        We also apply StandardScaler normalization because the player spy series in particular shows high variance.
        After training, predictions will be inverse-transformed to get the actual scale.
        
        If models for this table_index have already been trained, load them from the module-level cache.
        """
        self.table_index = table_index
        self.train_path = "train.csv"
        self.window_size = 5
        
        if table_index in _trained_models:
            # Retrieve the trained scalers and ARIMA parameters from cache
            (self.scaler_player, self.scaler_dealer, 
             self.player_arima_params, self.dealer_arima_params) = _trained_models[table_index]
        else:
            # Load training data for player and dealer streams
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
            
            # Save the original series for further use
            self.player_spy_series = player_spy
            self.dealer_spy_series = dealer_spy
            
            # Find best ARIMA parameters for player and dealer series
            print(f"Finding best ARIMA parameters for Table {table_index} Player series...")
            self.player_arima_params = find_best_arima_params(player_spy_scaled)
            print(f"Best ARIMA parameters for Player: {self.player_arima_params}")
            
            print(f"Finding best ARIMA parameters for Table {table_index} Dealer series...")
            self.dealer_arima_params = find_best_arima_params(dealer_spy_scaled)
            print(f"Best ARIMA parameters for Dealer: {self.dealer_arima_params}")
            
            # Store the ARIMA parameters and scalers in the module-level cache
            _trained_models[table_index] = (self.scaler_player, self.scaler_dealer, 
                                          self.player_arima_params, self.dealer_arima_params)
    
    def get_card_value_from_spy_value(self, value):
        """
        Given a spy value (float), return the corresponding card value.
        For this part we use a deterministic mapping using the Decision Tree model trained earlier.
        
        Output:
            A scalar integer prediction for the card value.
        """
        prediction = SPY_To_Val_MODEL.predict(np.array([[value]]))
        return int(prediction[0])
    
    def get_player_spy_prediction(self, hist):
        """
        Given a history (1D numpy array of length 5) of player spy values, use ARIMA to predict
        the next spy value.
        
        The history is first normalized using the player scaler, then fed to the model.
        The model's output is inverse-transformed to get the final prediction in the original scale.
        
        Output:
            A scalar float representing the predicted spy value.
        """
        # Normalize the history using the player's scaler
        hist = np.array(hist).reshape(-1, 1)
        hist_scaled = self.scaler_player.transform(hist).flatten()
        
        # Fit ARIMA model with the best parameters
        model = ARIMA(hist_scaled, order=self.player_arima_params)
        model_fit = model.fit()
        
        # Predict the next value
        forecast = model_fit.forecast(steps=1)
        prediction_scaled = forecast[0]
        
        # Inverse transform to get the prediction in original scale
        prediction = self.scaler_player.inverse_transform([[prediction_scaled]])[0,0]
        return prediction
    
    def get_dealer_spy_prediction(self, hist):
        """
        Given a history (1D numpy array of length 5) of dealer spy values, use ARIMA to predict
        the next spy value.
        
        The history is normalized using the dealer scaler, then fed to the model.
        The output is inverse-transformed to get the prediction in the original scale.
        
        Output:
            A scalar float representing the predicted spy value.
        """
        # Normalize the history using the dealer's scaler
        hist = np.array(hist).reshape(-1, 1)
        hist_scaled = self.scaler_dealer.transform(hist).flatten()
        
        # Fit ARIMA model with the best parameters
        model = ARIMA(hist_scaled, order=self.dealer_arima_params)
        model_fit = model.fit()
        
        # Predict the next value
        forecast = model_fit.forecast(steps=1)
        prediction_scaled = forecast[0]
        
        # Inverse transform to get the prediction in original scale
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
                        game_index,
                        ):
        """
        Arguments:
        curr_spy_history_player: list -> real number spy value series of player observed upto this point
        curr_spy_history_dealer: list -> real number spy value series of dealer observed upto this point
        curr_card_history_player: list -> integer series of player denoting value of cards observed upto this point
        curr_card_history_dealer: list -> integer series of dealer denoting value of cards observed upto this point
        curr_player_total: integer score of player
        curr_dealer_total: integer score of dealer
        turn: string -> either "player" or "dealer" denoting if its the player drawing right now or the dealer opening her cards
        game_index: integer -> tells which game is going on. Can be useful to figure if a new game has started

        Note that corresponding series of card and spy values are of the same length

        Output:
            if turn=="player" output either string "hit" or "stand" based on your decision
            else if turn=="dealer" output either string "surrender" or "continue" based on your decision
        """
        # Ensure we have enough history for prediction
        if len(curr_spy_history_player) >= self.window_size and len(curr_spy_history_dealer) >= self.window_size:
            # Predict next spy value and corresponding card value
            next_spy_player = self.get_player_spy_prediction(curr_spy_history_player[-self.window_size:])
            next_spy_dealer = self.get_dealer_spy_prediction(curr_spy_history_dealer[-self.window_size:])
            next_pred_card_player = self.get_card_value_from_spy_value(next_spy_player)
            next_pred_card_dealer = self.get_card_value_from_spy_value(next_spy_dealer)
            
            if turn == 'player':
                # Simulate taking another card
                potential_total = curr_player_total + next_pred_card_player
                
                # Basic strategy: stand at or above 17, or if taking another card would bust
                if potential_total >= 17 or potential_total > 21:
                    return 'stand'
                else:
                    return 'hit'
            else:  # dealer's turn
                # Predict what the dealer will get
                potential_dealer_total = curr_dealer_total + next_pred_card_dealer
                
                # Surrender if the dealer is likely to get a better hand
                if potential_dealer_total > curr_player_total and potential_dealer_total <= 21:
                    return 'surrender'
                else:
                    return 'continue'
        else:
            # Not enough history for prediction, use a simple strategy
            if turn == 'player':
                return 'hit' if curr_player_total < 17 else 'stand'
            else:
                # Without prediction, continue unless dealer already has a strong hand
                return 'surrender' if curr_dealer_total >= curr_player_total else 'continue'