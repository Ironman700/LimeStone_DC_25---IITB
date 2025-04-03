import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import tqdm
import PlayerMulti

def simulate_multi(list_player_data, list_dealer_data,player_obj, debug=False):

    assert len(list_player_data)==len(list_dealer_data)
    scoreboard_keys=['player_bust', 'surrender', 'dealer_bust', 'player_win', 'dealer_win', 'tie']
    scoreboard={}
    for k in scoreboard_keys:
        scoreboard[k]=[0 for _ in range(len(list_player_data))]

    list_pstream=[player_data[:,0] for player_data in list_player_data]
    list_dstream=[dealer_data[:,0] for dealer_data in list_dealer_data]

    list_pstream_card = [player_data[:,1] for player_data in list_player_data]
    list_dstream_card = [dealer_data[:,1] for dealer_data in list_dealer_data]

    history = []
    score = 0

    num_games = 200

    games_played=0
    list_player_idx=[20 for _ in list_pstream]
    list_dealer_idx=[20 for _ in list_dstream]

    while games_played<num_games:
        games_played+=1
        if debug:
            print("="*5, f"Game: {games_played}. player_idx: {list_player_idx}. dealer_idx: {list_dealer_idx}.", "="*5)


        curr_player_total = [0 for _ in range(len(list_pstream))]
        curr_dealer_total = [0 for _ in range(len(list_dstream))]
        active_player=[True for _ in range(len(list_pstream))]

        while True:
            if debug:
                print("="*5, "Player's Call", "="*5)

            list_action = player_obj.get_player_action_multi(
                                            copy.deepcopy([pstream[:player_idx] for pstream,player_idx in zip(list_pstream,list_player_idx)]),
                                            copy.deepcopy([dstream[:dealer_idx] for dstream,dealer_idx in zip(list_dstream,list_dealer_idx)]),
                                            copy.deepcopy([pstream_card[:player_idx] for pstream_card,player_idx in zip(list_pstream_card,list_player_idx)]),
                                            copy.deepcopy([dstream_card[:dealer_idx] for dstream_card,dealer_idx in zip(list_dstream_card,list_dealer_idx)]),
                                            copy.deepcopy(curr_player_total),
                                            copy.deepcopy(curr_dealer_total),
                                            'player',
                                            copy.deepcopy(active_player),
                                            games_played
                                           )
            assert len(list_action)==len(list_pstream)
            for aidx in range(len(list_action)):
                if not active_player[aidx]:
                    continue
                action=list_action[aidx]
                assert action in ['hit', 'stand']
                if action == 'hit':
                    actual_spy_value= list_pstream[aidx][list_player_idx[aidx]]
                    curr_card_value = list_pstream_card[aidx][list_player_idx[aidx]]

                    if debug:
                        print("Table index: ", aidx)
                        print("Actual Spy Value", actual_spy_value)
                        print("Actual Card Value", curr_card_value)
                        print("Curr Total", curr_player_total[aidx])
                        print("Total value for player:", curr_player_total[aidx] + curr_card_value)

                    curr_player_total[aidx] += curr_card_value
                    list_player_idx[aidx] += 1
                    if curr_player_total[aidx] > 21:
                        active_player[aidx]=False
                elif action == 'stand':
                    active_player[aidx]=False
                else:
                    raise

            if any(active_player):
                continue
            else:
                break
        if debug:
            print("Final player score: ", curr_player_total)

        active_dealer = [True for _ in (list_pstream)]

        for tidx in range(len(list_dstream)):
            if curr_player_total[tidx]>21:
                if debug:
                    print(f"Table {tidx} Bust.")
                scoreboard['player_bust'][tidx] += 1
                active_dealer[tidx]=False

        if not any(active_dealer):
            continue

        while True:
            if debug:
                print("="*5, "Dealer's Call", "="*5)

            list_action = player_obj.get_player_action_multi(copy.deepcopy([pstream[:player_idx] for pstream,player_idx in zip(list_pstream,list_player_idx)]),
                                            copy.deepcopy([dstream[:dealer_idx] for dstream,dealer_idx in zip(list_dstream,list_dealer_idx)]),
                                            copy.deepcopy([pstream_card[:player_idx] for pstream_card,player_idx in zip(list_pstream_card,list_player_idx)]),
                                            copy.deepcopy([dstream_card[:dealer_idx] for dstream_card,dealer_idx in zip(list_dstream_card,list_dealer_idx)]),
                                            copy.deepcopy(curr_player_total),
                                            copy.deepcopy(curr_dealer_total),
                                            'dealer',
                                            copy.deepcopy(active_dealer),
                                            games_played
                                           )
            assert len(list_action)==len(active_dealer)
            for aidx in range(len(active_dealer)):
                if not active_dealer[aidx]:
                    continue
                action=list_action[aidx]
                assert action in ['continue','surrender']
                if action == 'continue':
                    actual_spy_value=list_dstream[aidx][list_dealer_idx[aidx]]
                    curr_card_value = list_dstream_card[aidx][list_dealer_idx[aidx]]
                    if debug:
                        print("Table index: ", aidx)
                        print("Actual Spy Value", actual_spy_value)
                        print("Actual Card Value", curr_card_value)
                        print("Curr Total", curr_dealer_total[aidx])
                        print("Total value for dealer:", curr_dealer_total[aidx] + curr_card_value)

                    curr_dealer_total[aidx] += curr_card_value
                    list_dealer_idx[aidx] += 1
                    if curr_dealer_total[aidx] >= 17:
                        active_dealer[aidx]=False
                        assert curr_player_total[aidx]<=21, "wrong simulation, dealer is playing even though player has busted"

                        if curr_dealer_total[aidx]>21:
                            scoreboard['dealer_bust'][aidx] += 1
                        elif curr_dealer_total[aidx] > curr_player_total[aidx]:
                            scoreboard['dealer_win'][aidx] += 1
                        elif curr_dealer_total[aidx] < curr_player_total[aidx]:
                            scoreboard['player_win'][aidx] += 1
                        else:
                            scoreboard['tie'][aidx] += 1

                elif action == 'surrender':
                    active_dealer[aidx]=False
                    scoreboard['surrender'][aidx]+=1
                else:
                    raise
            if any(active_dealer):
                continue
            else:
                break

        if debug:
            print("Final dealer score: ", curr_dealer_total)

        score=0
        for aidx in range(len(list_pstream)):
            score+=(scoreboard['player_win'][aidx] + scoreboard['dealer_bust'][aidx]) - scoreboard['surrender'][aidx] * 0.5 - (scoreboard['player_bust'][aidx] + scoreboard['dealer_win'][aidx])

        if debug:
            print(scoreboard)
        history.append((score , copy.deepcopy(scoreboard)))

    if debug:
        print("scoreboard:", scoreboard)
        print("score:", score)

    score=0
    for aidx in range(len(list_pstream)):
        score+=(scoreboard['player_win'][aidx] + scoreboard['dealer_bust'][aidx]) - scoreboard['surrender'][aidx] * 0.5 - (scoreboard['player_bust'][aidx] + scoreboard['dealer_win'][aidx])

    print("scoreboard:", scoreboard)
    print("score:", score)
    return scoreboard, history,score

def score_game_multi(data_path,sicily_data_path,debug=False):
    player_obj=PlayerMulti.MyPlayerMulti()
    table_indices=player_obj.choose_tables()
    assert min(table_indices)>=0
    assert max(table_indices)<=4

    sicily_player=PlayerMulti.sicily_data_loader(sicily_data_path,'player')
    sicily_dealer=PlayerMulti.sicily_data_loader(sicily_data_path,'dealer')

    input_player_list=[PlayerMulti.data_loader(data_path,int(x),'player') for x in table_indices ]
    input_dealer_list=[PlayerMulti.data_loader(data_path,int(x),'dealer') for x in table_indices ]
    
    input_player_list.append(sicily_player)
    input_dealer_list.append(sicily_dealer)

    scoreboard,history,score=simulate_multi(input_player_list, input_dealer_list,player_obj,debug=debug)
    return score

score_game_multi('train.csv','sicilian_train.csv',False)
