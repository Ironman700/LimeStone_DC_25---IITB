import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import pandas as pd

def data_loader(path, table_idx, player_or_dealer):
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

def sicily_data_loader(path, player_or_dealer):
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'special_table', player_or_dealer, 'spy')]
    card = data[(f'special_table', player_or_dealer, 'card')]
    return np.array([spy, card]).T

class MyPlayerMulti:
    def __init__(self,lag=int(5)):
        pass
  
    def choose_tables(self):
        """
        return the indices of the tables you would like to play at. List of streams received in get_player_action_multi will be of the tables you provide below
        followed by the sicily table in the end

        The body is random for now, change accordingly
        Make sure that each of the index returned by this function can be loaded using the data_loader function above

        Return-> list[int] of the tables to play on
        
        """
        return [0,1]
    def get_player_action_multi(self,
                            list_curr_spy_history_player, 
                            list_curr_spy_history_dealer,
                            list_curr_card_history_player, 
                            list_curr_card_history_dealer, 
                            list_curr_player_total, 
                            list_curr_dealer_total, 
                            turn, 
                            active_tables,
                            game_index,
                            ):
        """
        Arguments:
        list_curr_spy_history_player: list[list] -> real number spy value series of player observed upto this point
        list_curr_spy_history_dealer: list[list] -> real number spy value series of dealer observed upto this point
        list_curr_card_history_player: list[list] -> integer series of player denoting value of cards observed upto this point
        list_curr_card_history_dealer: list[list] -> integer series of dealer denoting value of cards observed upto this point
        list_curr_player_total: list[int] -> list of integer score of player
        list_curr_dealer_total: list[int] -> list integer score of dealer
        turn: string -> either "player" or "dealer" denoting if its the player drawing right now or the dealer opening her cards
        active_tables: list[bool] -> tells on which tables we can make an action . If turn="player" then it is True for tables we have not busted at or not stood at
        If turn="dealer", then it is True on tables where you have not surrendered, and dealer is below 16.
        game_index: integer -> tells which game is going on. Can be useful to figure if a new game has started

        Note that correspopding lists of corresponding card and spy values are of the same length.  For eg the series  of table 1 player card and spy valus are of the same length.
        Each of the above series have the same length = number of tables you choose to play at + 1 (for the sicily table)
        
        The ordering is also same which you provided via the choose_tables function

        The body is random for now, change accordingly

        Output:
            if turn=="player" output a list of size len(active_tables) (this length is same as the length of all other list arguments as well). Each argument should be "stand" or "hit". The action
            for tables which have their entry for active_tables as False will be no-op.
            else if turn=="dealer" output a list of size len(active_tables) (this length is same as the length of all other list arguments as well). Each argument should be "continue" or "surrender". 
            The action for tables which have their entry for active_tables as False will be no-op.
        """
        if turn=='player':
            return ['stand','stand','stand']

        elif turn=='dealer':
            return ['surrender','surrender','surrender']