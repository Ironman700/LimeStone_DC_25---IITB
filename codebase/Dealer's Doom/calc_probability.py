import numpy as np
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import pandas as pd

def data_loader(path, table_idx, player_or_dealer):
    #utility for loading train.csv, example use in the notebook
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

def bust_probability(dealer_cards):
    """
    dealer_cards: list of integers representing the card values drawn from the dealer's deck
    output: probability that the dealer busts (i.e., goes over 21)
    
    The function simulates dealer runs based on the rule:
      - The dealer hits (draws a new card) while their total is 16 or less.
      - When the total exceeds 16, the dealer stands.
    If the dealer's total exceeds 21, the run is considered a bust.
    """
    total_runs = 0
    bust_runs = 0
    i = 0
    n = len(dealer_cards)
    
    while i < n:
        current_sum = 0
        
        # Start a new run and keep drawing until total > 16
        while i < n and current_sum <= 16:
            current_sum += dealer_cards[i]
            i += 1
        
        # Only count complete runs
        if current_sum > 16:
            total_runs += 1
            if current_sum > 21:
                bust_runs += 1
                
    # Return the bust probability
    return bust_runs / total_runs if total_runs > 0 else 0
