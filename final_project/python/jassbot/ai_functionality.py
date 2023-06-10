#!/usr/bin/env python3
# File name: ai_functionality.py
import logging
# imports
from random import choice
import pandas as pd
import itertools

import env
import function as f
import numpy as np
import config as c

# need for speed
# code that doesn't need to be run all the time in the class/methods
#  ---------------------------------------------------------------------
# file = "data/mcc_optimal_policy_100.csv"
# file = "mcc_optimal_policy_10000.csv"
# file = "sarsa_optimal_policy_1000.csv"
df = pd.read_csv(c.csv_file)
# ----------------------------------------------------------------------
s = ["1", "2", "3", "4"]
v = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
st = ["0", "1", "2", "3"]  # , "4"]
comb = []
for item in list(itertools.product(st, v, s)):
    comb.append(int(item[0] + item[1] + item[2]))
# ----------------------------------------------------------------------
# array_file = "/home/raphael/Development/JassBot/jassbot/mcc_max_Q_array_1000.npy"
# array_file = "/data/mcc_max_Q_array_100000.npy"
with open(c.npy_file, 'rb') as readfile:
    lookup = np.load(readfile)
# ----------------------------------------------------------------------
card_points = {'Rose-6': 0, 'Rose-7': 0, 'Rose-8': 0, 'Rose-9': 0,
               'Rose-Banner': 10, 'Rose-Under': 2, 'Rose-Ober': 3,
               'Rose-König': 4, 'Rose-Ass': 11, 'Eichel-6': 0, 'Eichel-7': 0,
               'Eichel-8': 0, 'Eichel-9': 0, 'Eichel-Banner': 10,
               'Eichel-Under': 2, 'Eichel-Ober': 3, 'Eichel-König': 4,
               'Eichel-Ass': 11, 'Schellen-6': 0, 'Schellen-7': 0,
               'Schellen-8': 0, 'Schellen-9': 0, 'Schellen-Banner': 10,
               'Schellen-Under': 2, 'Schellen-Ober': 3, 'Schellen-König': 4,
               'Schellen-Ass': 11, 'Schilten-6': 0, 'Schilten-7': 0,
               'Schilten-8': 0, 'Schilten-9': 0, 'Schilten-Banner': 10,
               'Schilten-Under': 2, 'Schilten-Ober': 3, 'Schilten-König': 4,
               'Schilten-Ass': 11, 'Trump-6': 0, 'Trump-7': 0, 'Trump-8': 0,
               'Trump-9': 14, 'Trump-Banner': 10, 'Trump-Under': 20,
               'Trump-Ober': 3, 'Trump-König': 4, 'Trump-Ass': 11}


class AI:
    """
    Provides all methods to decide what card to play
    """

    def __init__(self, agent, player, trump, played, card_class):
        self.player = player
        self.trump = trump
        self.played = played
        self.agent = agent
        self.card_class = card_class

    def run(self):
        """
        Depending on the agent we run a different methode

        :return: the card we play
        """

        match self.agent:
            case "random":
                return self.random()
            case "take":
                return self.always_strongest()
            case "leave":
                return self.always_weakest()
            case "always_trump":
                return self.always_trump()
            case "always_strongest":
                return self.always_strongest()
            case "optimal_policy":
                return self.optimal_policy()
            case "op":
                return self.op()
            case "statistics":
                return self.statistics()
            case "probabilistic":
                return self.probabilistic()
            case "rules":
                return self.rules()
            case "human":
                self.human()

    def random(self):
        """
        Choose a random card to play

        :return: card
        """
        return choice(self.player.playable_cards)

    def always_trump(self):
        """
        Always play a random trump card if possible

        :return: card
        """
        play = []
        for card in self.player.playable_cards:
            if card.suit == self.trump.suit:
                play.append(card)

        if not play:
            return choice(self.player.playable_cards)
        else:
            return choice(play)

    def always_strongest(self):
        """
        Always play the strongest card if possible

        :return: card
        """
        play = []

        if self.played is not None:
            for card in self.player.hand:  # self.player.playable_cards:
                if f.strength(card, self.trump) > f.strength(self.played, self.trump) and card.suit != self.trump.suit:
                    play.append(card)
        if not play:
            # return choice(self.player.playable_cards)
            return max(self.player.playable_cards)
        else:
            # return choice(play)
            return min(play)

    def always_weakest(self):
        """
        Always play the weakest card

        :return: card
        """
        play = []
        if self.played is not None:
            for card in self.player.hand:  # self.player.playable_cards:
                if f.strength(card, self.trump) < f.strength(self.played, self.trump) and card.suit != self.trump.suit:
                    play.append(card)

        if not play:
            # return choice(self.player.playable_cards)
            return min(self.player.playable_cards)
        else:
            # return choice(play)
            return max(play)

    def optimal_policy(self):
        """
        Play card depending on the optimal policy (mcc/sarsa), chose source
        file on top of the module
        This method uses a csv file with pandas and a good amount of compute
        time therefor the op methode has been created to speed things up

        Advised not to use...

        :return: card
        """

        if self.played is None:
            # we play the first card (random)
            # return choice(self.player.playable_cards)
            return self.always_strongest()
        else:
            # we play the second card
            # determine card number of the played card and look it up in the
            # optimal policy file
            played_card_number = str(((f.card_to_number(self.played, self.trump, None))))
            # print(played_card_number)
            optimal_cards_df = df[["player1", played_card_number]]
            optimal_cards_df = optimal_cards_df[optimal_cards_df[played_card_number] == "take"]
            optimal_card_numbers = optimal_cards_df["player1"].tolist()

            optimal_cards = []

            # print(f"playable cards: {self.player.playable_cards}")
            for optimal_card_number in optimal_card_numbers:
                # print(optimal_card_number)
                op_card = comb[optimal_card_number]
                # print("opot_card: ", op_card)
                if len(str(op_card)) == 1:
                    s = op_card - 1
                    v = 6
                elif op_card > 179:
                    pass
                else:
                    s, v = int(str(op_card)[-1]) - 1, int(str(op_card)[-2]) + 6
                # print("v, s: ", v, s)
                card = self.card_class(v, s)
                # print(card, type(card))
                # print(self.player.playable_cards[1], type(self.player.playable_cards[1]))

                if str(card) in str(self.player.playable_cards):
                    if not str(card) in str(optimal_cards):
                        optimal_cards.append(card)

            if not optimal_cards:
                return self.always_strongest()
            else:
                # return choice(optimal_cards)
                return min(optimal_cards)

    def op(self):
        """
        Play card depending on the optimal policy (mcc/sarsa), chose source
        file on top of the module
        This method uses a npy file with numpy and is an optimized version of
        the optimal policy methode

        :return: card
        """

        if self.played is None:
            # we play the first card (random)
            # return choice(self.player.playable_cards)
            return self.always_strongest()
        else:
            # we play the second card
            # determine card number of the played card and look it up in the
            played_card_number = str(((f.card_to_number(self.played, self.trump, None))))

            # show column
            col = lookup[int(played_card_number), :]
            if col.max() > 0:
                # if there are values bigger than 0

                # get the row numbers where the values are maximal
                row = np.where(col == col.max())

                # chose one row (card) to play
                op_card = c.comb[choice(row[0])]

                s, v = f.number_to_card(str(op_card))
                card = self.card_class(v, s)

                if str(card) in str(self.player.playable_cards):
                    optimal_card = card
                    # print("---------opt card: ", optimal_card)
                    return optimal_card
                else:
                    # return choice(self.player.playable_cards)
                    return self.always_strongest()
            else:
                # return choice(self.player.playable_cards)
                return self.always_strongest()

    def statistics(self):
        # print(f.cardlist_to_text(self.player.hand))
        # print(f.cardlist_to_text(self.player.playable_cards))
        # print(self.played)
        # print(self.trump)

        data = []

        h = len(self.player.hand)
        p = self.played
        t = 1
        g = 0 + t
        d = 36  # number of cards in the deck
        P = 0.
        W = 0
        nf = 0

        if self.played is None:
            return self.total_statistics()
            # stp = -1
            # pp = 0
        else:
            stp = f.strength(self.played, self.trump)
            pp = f.points(self.played, self.trump)

            def kp():
                kp = 0
                for k in self.player.hand:
                    if f.strength(k, self.trump) > stp:
                        kp += 1
                return kp

            for n, k in enumerate(self.player.playable_cards):
                nf += nf + h - kp()
                tp = d - h - g - t - n
                P = nf / tp
                if f.strength(k, self.trump) > stp:
                    W = f.points(k, self.trump) + pp

                data.append([k, P, W, P * W])

            ndata = np.array(data)
            # print(f"card, probability, points:\n{ndata}")
            # print(f"shape: {ndata.shape}")

            pre_card = np.where(ndata == np.max(ndata[:, -1]))

            if ndata[pre_card[0][0], 0] is None:
                # return choice(self.player.playable_cards)
                return self.always_strongest()
            else:
                return ndata[pre_card[0][0], 0]

    def total_statistics(self):
        # print(f.cardlist_to_text(self.player.hand))
        # print(f.cardlist_to_text(self.player.playable_cards))
        # print(self.played)
        # print(self.trump)

        data = []

        h = len(self.player.hand)
        p = self.played
        t = 1
        g = 0 + t
        d = 36  # number of cards in the deck
        P = 0.
        W = 0
        nf = 0
        played = []
        for card in env.Deck().cards:
            if card not in [self.player.hand, self.trump]:
                played.append(card)

        def kp():
            kp = 0
            for k in self.player.hand:
                if f.strength(k, self.trump) > stp:
                    kp += 1
            return kp

        for nn, kk in enumerate(self.player.hand):
            stp = f.strength(kk, self.trump)
            pp = f.points(kk, self.trump)
            for n, k in enumerate(list(played)):
                nf += nf + h - kp()
                tp = d - h - g - t - nn
                P += nf / tp
                if f.strength(k, self.trump) > stp:
                    W += f.points(k, self.trump) + pp

            data.append([kk, P, W, P * W])

        ndata = np.array(data)
        # print(f"card, probability, points:\n{ndata}")
        # print(f"shape: {ndata.shape}")

        pre_card = np.where(ndata == np.max(ndata[:, -1]))

        if ndata[pre_card[0][0], 0] is None:
            print("play random - statistics says: ", ndata[pre_card[0][0], 0])
            # return choice(self.player.playable_cards)
            return self.always_strongest()
        else:

            return ndata[pre_card[0][0], 0]

    def probabilistic(self):
        played = []
        hand = list(map(str, self.player.hand))
        hand = list(map(str, self.player.playable_cards))
        # print(hand)
        trump = str(self.trump)
        deck = list(map(str, env.Deck().cards))

        def strength(card, trump):
            """
            Gives back the strength of the card - the higher, the better

            :param card: card for witch the strength needs to be determent
            :param trump: trump suit
            :return: integer value of the strength
            """
            cs = card.split("-")[0]
            cv = card.split("-")[-1]
            ts = trump.split("-")[0]

            if cv == "Ass":
                cv = 14
            elif cv == "König":
                cv = 13
            elif cv == "Ober":
                cv = 12
            elif cv == "Under":
                cv = 11
            elif cv == "Banner":
                cv = 10

            if cs == ts:
                # if nell
                if cv == 9:
                    st = 19
                # if buur
                elif cv == 11:
                    st = 18  # 30
                else:
                    st = int(cv) + 3
            else:
                st = int(cv) - 6
            return st

        def unknown_cards(hand, played, trump, deck):
            if self.played is None:
                # print(set(deck) - set(played) - set(trump) - set(hand))
                # print(type(set(deck) - set(played) - set(trump) - set(hand)))
                return set(deck) - set(played) - set(trump) - set(list(map(str, self.player.hand)))
            else:
                # print("played...")
                return set(deck) - set(played) - set(trump) - set(list(map(str, self.player.hand))) - set(str(self.played))

        def probability(favorable, possible):
            return len(favorable) / len(possible)

        if self.played is None:
            # if we are first to play
            lookup_point_probability = {}
            point_probability = {}
            for i in range(0, len(hand)):
                for h in range(0, len(hand)):
                    # print(f"i: {i}")
                    # print(f"h: {h}")
                    # print(f"len hand: {len(hand)}")
                    if i + h < len(hand):
                        idx = h + i
                    else:
                        idx = h + i - len(hand) - 1
                    # print(f"index: {idx}\n--------------------------------------------------------")
                    played = []
                    pp_sum = 0.
                    for unknown_card in unknown_cards(hand, played, trump, deck):
                        # check if h beats unknown card
                        if strength(hand[idx], trump) >= strength(unknown_card, trump):
                            p = probability(unknown_card, unknown_cards(hand, played, trump, deck))
                            prob_point = p * (card_points[hand[idx]] + card_points[unknown_card])
                            point_probability[hand[idx], unknown_card] = prob_point
                            pp_sum += prob_point
                            played.append(unknown_card)
                    # add up "probability points"
                    lookup_point_probability[hand[i]] = point_probability, pp_sum

            # print(point_probability)
            # m = max(point_probability.values())
            # print(m)
            # print("Play: ", list(point_probability.keys())[list(point_probability.values()).index(m)][0])
            # print(lookup_point_probability)
            card_to_play = "", 0.
            for card_in_hand, prob in lookup_point_probability.items():
                # print(f"{card_in_hand}: {prob}")
                # print(f"first played: {card_in_hand}")
                # print(f"sum of point probabilitys: {prob[1]:0.0f}\n----------------------------------------")
                if prob[1] > card_to_play[1]:
                    card_to_play = card_in_hand, prob[1]

            # m = list(list(lookup_point_probability.values()))[1]
            # print(card_to_play)
            for card in self.player.hand:
                # print(card, type(card))
                # print(str(card), type(str(card)))
                # print(card_to_play, type(card_to_play))
                if str(card) == card_to_play[0]:
                    # print(card, type(card))
                    return card

        else:
            # hand = list(map(str, self.player.playable_cards))
            # print("-----")
            # print(f"hand: {hand}")
            # print(f"trump: {trump}")

            # if we play second
            lookup_point_probability = {}
            point_probability = {}
            first_played = str(self.played)
            # print(f"first played: {first_played}")

            for i in range(0, len(hand)):
                for h in range(0, len(hand)):
                    # print(f"i: {i}")
                    # print(f"h: {h}")
                    # print(f"len hand: {len(hand)}")
                    if i + h < len(hand):
                        idx = h + i
                    else:
                        idx = h + i - len(hand) - 1
                    # print(f"index: {idx}\n--------------------------------------------------------")
                    played = []
                    pp_sum = 0.
                    if strength(hand[idx], trump) >= strength(first_played, trump):
                        p = probability(first_played, unknown_cards(hand, played, trump, deck))
                        prob_point = p * (card_points[hand[idx]] + card_points[first_played])
                        point_probability[hand[idx], first_played] = prob_point
                        pp_sum += prob_point
                        played.append(first_played)
                # add up "probability points"
                lookup_point_probability[hand[i]] = point_probability, pp_sum

            for i in range(0, len(hand)):
                for h in range(1, len(hand)):
                    # print(f"i: {i}")
                    # print(f"h: {h}")
                    # print(f"len hand: {len(hand)}")
                    if i + h < len(hand):
                        idx = h + i
                    else:
                        idx = h + i - len(hand) - 1
                    # print(f"index: {idx}\n--------------------------------------------------------")
                    played = []
                    pp_sum = 0.
                    for unknown_card in unknown_cards(hand, played, trump, deck):
                        # check if h beats unknown card
                        if strength(hand[idx], trump) >= strength(unknown_card, trump):
                            p = probability(unknown_card, unknown_cards(hand, played, trump, deck))
                            prob_point = p * (card_points[hand[idx]] + card_points[unknown_card])
                            point_probability[hand[idx], unknown_card] = prob_point
                            pp_sum += prob_point
                            played.append(unknown_card)
                    # add up "probability points"
                    lookup_point_probability[hand[i]] = point_probability, pp_sum

            # print(point_probability)
            # m = max(point_probability.values())
            # print(m)
            # print("Play: ", list(point_probability.keys())[list(point_probability.values()).index(m)][0])
            # print(lookup_point_probability)
            card_to_play = "", 0.

            for card_in_hand, prob in lookup_point_probability.items():
                if prob[1] > card_to_play[1]:
                    card_to_play = card_in_hand, prob[1]

            # m = list(list(lookup_point_probability.values()))[1]
            # print(card_to_play)
            for card in self.player.hand:
                # print(card, type(card))
                # print(str(card), type(str(card)))
                # print(card_to_play, type(card_to_play))
                if str(card) == card_to_play[0]:
                    # print(card, type(card))
                    return card

        # card = choice(self.player.playable_cards)
        card = self.always_strongest()
        return card

    def rules(self):
        pass

    def human(self):
        print("Human")

        return choice(self.player.playable_cards)


def main():
    """
    Tests for the file, does it run?
    """

    import env
    deck = env.Deck()
    p1 = env.Player("player1")
    trump = deck.cards[-1]
    played = deck.cards[-2]
    p1.hand = deck.cards[0:9]
    p1.playable_cards = p1.hand[0:4]
    p1.played_cards = None  # played

    ai = AI("probabilistic", p1, trump, played, env.Card)
    # c = ai.op()
    c = ai.probabilistic()
    print(c)


if __name__ == '__main__':
    for i in range(0, 1):
        print(f">>>>>>>>>>>>>>>  {i}  <<<<<<<<<<<<<<<")
        main()
