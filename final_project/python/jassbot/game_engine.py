#!/usr/bin/env python3
# File name: game_engine.py
import copy
# imports
from random import shuffle, choice
import time
import logging

import ai_functionality


# agents random, trump, strong, human
AGENT1 = "random"
AGENT2 = "random"  # "probabilistic"

# logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logging.basicConfig(filename='app.log', filemode='w', format='%(message)s',
                    level=logging.WARNING)


def points(card, trump):
    # points if not trump
    points = {6: 0, 7: 0, 8: 0, 9: 0, 10: 10, 11: 2, 12: 3, 13: 4, 14: 11}

    # check if trump card
    if card.suit == trump.suit:
        # if nell
        if card.value == 9:
            point = 14
        # if buur
        elif card.value == 11:
            point = 20
        else:
            point = points[card.value]
    else:
        point = points[card.value]
    return point


def strength(card, trump):
    if card.suit == trump.suit:
        # if nell
        if card.value == 9:
            strength = 29
        # if buur
        elif card.value == 11:
            strength = 30
        else:
            strength = 14 + card.value
    else:
        strength = card.value
    return strength


class Card:
    """
    Definition of the cards

    """
    suits = ["Rose", "Eichel", "Schellen", "Schilten"]

    values = [None, None, None, None, None, None, "6", "7", "8", "9", "Banner",
              "Under", "Ober", "König", "Ass"]

    def __init__(self, v, s):
        """suit + value are inits"""
        self.value = v
        self.suit = s
        self.point = None
        self.strength = None

    def __lt__(self, c2):
        """lesser than"""
        if self.value < c2.value:
            return True
        if self.value == c2.value:
            if self.suit < c2.suit:
                return True
            else:
                return False
        return False

    def __gt__(self, c2):
        """greater than"""
        if self.value > c2.value:
            return True
        if self.value == c2.value:
            if self.suit > c2.suit:
                return True
            else:
                return False
        return False

    # def __repr__(self):
    #     v = self.suits[self.suit] + "-" + self.values[self.value]
    #     return v

    def __str__(self):
        v = self.suits[self.suit] + "-" + self.values[self.value]
        return v


class Deck:
    """
    Deck creation

    """

    def __init__(self):
        self.cards = []
        for i in range(6, 15):
            for j in range(4):
                self.cards.append(Card(i, j))
        shuffle(self.cards)


class Player:
    def __init__(self, name):
        self.points = 0
        self.hand = []
        self.playable_cards = []
        self.played_cards = []
        self.name = name


class Game:
    def __init__(self):
        # player1 = input("p1 name ")
        # player2 = input("p2 name ")
        player1 = "Player 1"
        player2 = "Player 2"
        self.p1 = Player(player1)
        self.p2 = Player(player2)
        self.trump = None
        self.data = []

        # random choose dealer at start of the game
        self.players = [self.p1, self.p2]
        self.dealer = choice(self.players)

    def collect_data(self, trump, p1, p2):
        if p1[1] > p2[1]:
            p1_win = 1
            p2_win = 0
        elif p1[1] < p2[1]:
            p1_win = 0
            p2_win = 1
        else:
            p1_win = 1
            p2_win = 1

        self.data.append([trump, p1[0], p1[1], p1_win])
        self.data.append([trump, p2[0], p2[1], p2_win])

    def play_game(self):
        deck = Deck()

        # reset points & co
        for player in self.players:
            player.points = 0
            player.hand = []
            player.playable_cards = []
            player.played_cards = []

        # deal cards
        cards = deck.cards
        logging.debug(f"Deck: {cards}")

        self.p1.hand = cards[0:9]
        self.p2.hand = cards[9:18]

        h1 = cards[0:9]
        h2 = cards[9:18]

        # logging.info(f"{self.p1.name}: Hand: {self.p1.hand}")
        # logging.info(f"{self.p2.name}: Hand: {self.p2.hand}")

        # define trump
        self.trump = cards[18]
        # logging.info(f"Trump: {self.trump}")

        data = []

        # play rounds -----------------------------------------------------
        for current_round in range(0, 9):  # 9 is the good number
            # logging.info(f"\nPlaying Round {current_round}")
            self.play_round(self.trump)  # -------------------------------
            # update state and action
            pass

        # logging.info(f"\nResults")
        # logging.info(f"Points for {self.p1.name}: {self.p1.points}")
        # logging.info(f"Points for {self.p2.name}: {self.p2.points}")

        self.collect_data(self.trump, [h1, self.p1.points],
                          [h2, self.p2.points])

        return [copy.copy(self.p1), copy.copy(self.p2)]

    def play_round(self, trump):
        first = self.dealer
        for player in self.players:
            if player.name == self.dealer.name:
                pass
            else:
                second = player

        # first player plays card
        played1, first.hand, first.played_cards = self.play_card(first, trump)
        # second player plays card
        played2, second.hand, second.played_cards = self.play_card(second,
                                                                   trump,
                                                                   played1)

        # logging.info(f"trump: {trump}")
        # logging.info(f"{first.name} played card: {played1}")
        # logging.info(f"{second.name} played card: {played2}")

        if strength(played1, trump) > strength(played2, trump):
            first.points = first.points + points(played1, trump)
            self.dealer = first
            # logging.info(f"{first.name} winns this round")
        else:
            second.points = second.points + points(played2, trump)
            self.dealer = second
            # logging.info(f"{second.name} winns this round")

    def play_card(self, player, trump, played=None):
        if player.name == "Player 1":
            agent = AGENT1
        else:
            agent = AGENT2

        # if nothing is played, yet we are free to choose whatever we want
        if played is None:
            player.playable_cards = player.hand
        else:
            # figure out what cards can be played
            player.playable_cards = self.playable_cards(player.hand, trump,
                                                        played)

        ai = ai_functionality.AI(agent, player, trump, played, Card)
        # if only one card is left we have to play it
        if len(player.playable_cards) == 1:
            played = player.playable_cards[0]

        else:
            # AI method hook

            played = ai.run()
            # print("player: ", player.name)
            # print("played: ", played, type(played))
            # print("player hand: ", player.hand)
            player.hand.remove(played)
            player.played_cards.append(played)
        # print(played)
        return played, player.hand, player.played_cards

    def playable_cards(self, hand, trump, played):
        playable_cards = []
        for card in hand:
            if card.suit == played.suit:
                playable_cards.append(card)
            elif card.suit == trump.suit:
                if card.value > trump.value:
                    playable_cards.append(card)
        if playable_cards == []:
            playable_cards = hand
        return playable_cards


def main():
    import pandas as pd
    game = Game()
    d = []
    for i in range(0, 100_000_000):
        game.play_game()
        # print(game.data.__str__())
    print("gathering complet")
    for i in range(0, len(game.data), 2):
        # print(game.data[i][1])
        d.append(game.data[i][1])
    # print(d)
    print("creating dataframe")
    df = pd.DataFrame(d, columns=["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"])
    # print(df)
    print("write data to disk")
    df.to_csv('hands.csv', index=True)


if __name__ == '__main__':
    main()
