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

def simulate(player_data, dealer_data,player,num_games=200, debug=False):
    pstream=player_data[:,0]
    dstream=dealer_data[:,0]
    pstream_card=player_data[:,1]
    dstream_card=dealer_data[:,1]
    
    scoreboard = {'player_bust':0, 'surrender':0, 'dealer_bust':0, 'player_win':0, 'dealer_win':0, 'tie':0}
    history = []
    player_idx = 20
    dealer_idx = 20
    games = 0
    pbar=tqdm.tqdm(total=num_games)
    while player_idx < len(pstream) and dealer_idx < len(dstream):
        if debug:
            print("="*5, f"Game: {games}. player_idx: {player_idx}. dealer_idx: {dealer_idx}.", "="*5)
        if games == num_games:
            break
        games += 1
        pbar.update(1)
        
        curr_player_total = 0
        curr_dealer_total = 0
        
        while True:
            if debug:
                print("="*5, "Player's Call", "="*5)
                print("player spy", pstream[player_idx-10:player_idx])
            action = player.get_player_action(pstream[:player_idx], dstream[:dealer_idx], pstream_card[:player_idx], dstream_card[:dealer_idx], curr_player_total, curr_dealer_total, 'player', games)

            assert action in ['hit', 'stand']
            
            if action == 'hit':
                curr_card_value = pstream_card[player_idx]
                
                if debug:
                    print("Actual Spy Value", pstream[player_idx])
                    print("Actual Card Value", curr_card_value)
                    print("Curr Total", curr_player_total)
                    print("Total value for player:", curr_player_total + curr_card_value)

                curr_player_total += curr_card_value
                player_idx += 1
                if curr_player_total > 21:
                    break
            elif action == 'stand':
                break
            else:
                raise
        if debug:
            print("Final player score: ", curr_player_total)
        surrender = 0
        
        if curr_player_total > 21:
            if debug:
                print("Bust.")
            scoreboard['player_bust'] += 1
            continue

        else:
            while True:
                if debug:
                    print("="*5, "Dealer's Call", "="*5)
                    print("dealer spy", dstream[dealer_idx-10:dealer_idx])
                action = player.get_player_action(pstream[:player_idx], dstream[:dealer_idx], pstream_card[:player_idx], dstream_card[:dealer_idx], curr_player_total, curr_dealer_total, 'dealer',games)
                if action == 'continue':
                    curr_card_value = dstream_card[dealer_idx]
                    
                    if debug:
                        print("Actual Spy Value", dstream[dealer_idx])
                        print("Actual Card Value", curr_card_value)
                        print("Curr Total", curr_dealer_total)
                        print("Total value for dealer:", curr_dealer_total + curr_card_value)

                    curr_dealer_total += curr_card_value
                    dealer_idx += 1
                    if curr_dealer_total >= 17:
                        break
                elif action == 'surrender':
                    surrender = 1
                    break

                else:
                    raise

        if debug:
            print("Final dealer score: ", curr_dealer_total)

        if surrender == 1:
            scoreboard['surrender'] += 1
        elif curr_dealer_total > 21:
            scoreboard['dealer_bust'] += 1
        else:
            if curr_dealer_total > curr_player_total:
                scoreboard['dealer_win'] += 1
            elif curr_dealer_total < curr_player_total:
                scoreboard['player_win'] += 1
            else:
                scoreboard['tie'] += 1
        
        score = (scoreboard['player_win'] + scoreboard['dealer_bust']) - scoreboard['surrender'] * 0.5 - (scoreboard['player_bust'] + scoreboard['dealer_win'])
        history.append((score, copy.deepcopy(scoreboard)))
        if debug:
            print("scoreboard:", scoreboard)
            print("score:", score)
            
    return scoreboard, history

def score_game(player_data,dealer_data,player,num_games=200,debug=False):
    scoreboard,history=simulate(player_data, dealer_data,player,num_games=num_games,debug=debug)
    score=(scoreboard['player_win'] + scoreboard['dealer_bust']) - scoreboard['surrender'] * 0.5 - (scoreboard['player_bust'] + scoreboard['dealer_win'])
    print(scoreboard)
    return score

if __name__=='__main__':
    table_index=4
    player=Player.MyPlayer(table_index)
    player_data=Player.data_loader("train.csv",table_index,'player')
    dealer_data=Player.data_loader("train.csv",table_index,'dealer')
    print("earnings")
    print(score_game(player_data,dealer_data,player,debug=False))