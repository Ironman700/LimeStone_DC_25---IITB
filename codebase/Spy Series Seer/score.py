import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import tqdm
import Player 


def score_mse(test_data,mode,player):
    test_data=test_data[:,0]

    preds=[]
    trues=[]
    for idx in range(5,len(test_data)):
        inp=test_data[idx-5:idx]
        if mode=='player':
            op=player.get_player_spy_prediction(inp)
        elif mode=='dealer':
            op=player.get_dealer_spy_prediction(inp)
        else:
            assert False, "mode unknown"
        preds.append(op)
        trues.append(test_data[idx])

    mse=np.mean((np.array(trues)-np.array(preds))**2)
    return mse

if __name__=='__main__':
    table_index=2
    player=Player.MyPlayer(table_index)
    player_data=Player.data_loader("train.csv",table_index,'player')
    dealer_data=Player.data_loader("train.csv",table_index,'dealer')
    print("player mse")
    print(score_mse(player_data,'player',player))
    print("dealer mse")
    print(score_mse(dealer_data,'dealer',player))
