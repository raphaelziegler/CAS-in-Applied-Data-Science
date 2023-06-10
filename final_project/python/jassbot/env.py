#!/usr/bin/env python3
# File name: env.py

# imports
from random import shuffle, choice
import logging
import function as f
import ai_functionality as ai_f

# Global variables
MIN_POINTS = 25  # minimal points to gain a point
LANG = "ch"  # card suits, fall back english card suites only ch is implemented

# set up for logging
logging.basicConfig(filename="../app.log", filemode="w",
                    format="%(asctime)s - %(message)s (%(levelname)s)",
                    datefmt="%y-%m-%d %H:%M:%S", level=logging.INFO)


class Card:
    """
    Definition of the cards,
    Swiss-German (ch)
    """

    # do not change, only swiss-german cards are implemented
    if LANG == "ch":
        suits = ["Rose", "Eichel", "Schellen", "Schilten"]
        values = [None, None, None, None, None, None, "6", "7", "8", "9", "Banner",
                  "Under", "Ober", "König", "Ass"]
    else:
        suits = ["Diamond", "Hearts", "Spades", "Clubs"]
        values = [None, None, None, None, None, None, "6", "7", "8", "9", "10",
                  "Jack", "Queen", "King", "Ace"]

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

    # makes problems, so I'm not using it
    # def __repr__(self):
    #     v = f"{self.suits[self.suit]}-{self.values[self.value]}"
    #     return v

    def __str__(self):
        v = f"{self.suits[self.suit]}-{self.values[self.value]}"
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
    """
    Player creation
    """
    def __init__(self, name):
        self.points = 0
        self.hand = []
        self.playable_cards = []
        self.played_cards = []
        self.name = name
        self.action = ""
        self.reward = 0
        self.wins = 0


class Handjass:
    """
    Provides all methods to play a game of Handjass
    """

    def __init__(self):
        # do not change the player names, they are hard coded in other modules
        player1 = "player1"
        player2 = "player2"
        self.p1 = Player(player1)
        self.p2 = Player(player2)
        self.trump = None
        self.game_state = 0

        # <<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.state = {player1: None, player2: None}  # initial state
        self.actions = ("take", "leave")  # actions that can be taken
        self.current_round = 0  # row counter (as we can not use a for loop)

        self.state = {self.p1.name: 0, self.p2.name: 0}
        init_state = self.state.copy()
        self.history = [init_state]

        # random choose dealer at start of the game
        self.players = [self.p1, self.p2]
        self.dealer = choice(self.players)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def game(self, state, p1_action, p2_action):
        """
        Play games until a player reaches 7 points
        :return: winner
        """

        if self.p1.wins == 7:
            # player 1 wins the Handjass game
            # logging.debug(f"Player 1 wins the Handjass game {self.p1.wins}/{self.p2.wins}")
            state = "terminal"
            reward = 1
            return state, reward
        elif self.p2.wins == 7:
            # player 2 wins the Handjass game
            # logging.debug(f"Player 2 wins the Handjass game {self.p2.wins}/{self.p1.wins}")
            state = "terminal"
            reward = -1
            return state, reward
        else:
            # no winner yet
            return self.play_game(state, p1_action, p2_action)

    def play_game(self, state, p1_action, p2_action):
        """
        Play the given hand

        :param state: state for reinforcement learning
        :param p1_action: chosen action for player 1
        :param p2_action: chosen action for player 2
        :return: state and reward for player 1
        """
        self.p1.action = p1_action
        self.p2.action = p2_action

        # instead of a for loop I use the mach control, so with each methode
        # call the action can be changed
        match self.current_round:
            case 0:
                # get deck
                deck = Deck()

                # deal cards
                cards = deck.cards
                logging.debug(f"Deck: {cards}")

                self.p1.hand = cards[0:9]
                self.p2.hand = cards[9:18]

                # logging.debug(f"{self.p1.name}: Hand: {self.p1.hand}")
                # logging.debug(f"{self.p2.name}: Hand: {self.p2.hand}")

                # define trump
                self.trump = cards[18]
                # logging.debug(f"Trump: {self.trump}")

                # play 1st card
                state, reward = self.play_round(self.trump)
                self.current_round += 1
                return state, reward
            case 1:
                # play 2nd card
                state, reward = self.play_round(self.trump)
                self.current_round += 1
                return state, reward
            case 2:
                # play 3rd card
                state, reward = self.play_round(self.trump)
                self.current_round += 1
                return state, reward
            case 3:
                # play 4th card
                state, reward = self.play_round(self.trump)
                self.current_round += 1
                return state, reward
            case 4:
                # play 5th card
                state, reward = self.play_round(self.trump)
                self.current_round += 1
                return state, reward
            case 5:
                # play 6th card
                state, reward = self.play_round(self.trump)
                self.current_round += 1
                return state, reward
            case 6:
                # play 7th card
                state, reward = self.play_round(self.trump)
                self.current_round += 1
                return state, reward
            case 7:
                # play 8th card
                state, reward = self.play_round(self.trump)
                self.current_round += 1
                return state, reward
            case 8:
                # play last / 9th card
                state, reward = self.play_round(self.trump)

                # logging.debug(f"Points for {self.p1.name}: {self.p1.points}")
                # logging.debug(f"Points for {self.p2.name}: {self.p2.points}")

                if self.p1.points <= MIN_POINTS:
                    self.p1.wins += -1
                if self.p2.points <= MIN_POINTS:
                    self.p2.wins += -1

                if self.p1.points > MIN_POINTS and self.p1.points > self.p2.points:
                    # player 1 wins this game
                    self.p1.wins += 1
                    self.p1.reward = 1
                    self.p2.reward = -1
                    # state = "terminal"
                    # new_state = state.copy()
                    # self.history.append(new_state)
                    return state, self.p1.reward
                elif self.p2.points > MIN_POINTS and self.p2.points > self.p1.points:
                    # player 2 wins this game
                    self.p2.wins += 1
                    self.p1.reward = -1
                    self.p2.reward = 1
                    # state = "terminal"
                    # self.history.append(state)
                    # return state, self.p1.reward
                else:
                    self.p1.reward = 0
                    self.p2.reward = 0
                    # state = "terminal"
                    # self.history.append(state)
                    # return state, self.p1.reward

                self.current_round = 0  # row counter (as we can not use a for loop)
                return state, reward

        # logging.error("#001 - This code should never be reached")
        reward = 0

        return state, reward

    def play_round(self, trump):
        """
        Each player plays a card from their hand

        :param trump: trump suit
        :return: state and reward for player 1
        """
        first = self.dealer
        for player in self.players:
            if player.name == self.dealer.name:
                pass
            else:
                second = player

        # first player plays card
        played1, first.hand, first.played_cards = self.play_card(first, trump)

        # second player plays card
        second.playable_cards = self.playable_cards(second.hand, trump, played1)
        played2, second.hand, second.played_cards = self.play_card(second, trump, played1)

        # check if we are allowed to play that card
        # print(played2, second.name, "-----------------------------------------")
        if played2 not in second.hand:
            state = "terminal"
            reward = -1
            return state, reward
        else:

            # logging.debug(f">>> trump: {trump}")
            # logging.debug(f"{first.name} played card: {played1}")
            # logging.debug(f"{second.name} played card: {played2}")

            if f.strength(played1, trump) > f.strength(played2, trump):
                # first player wins
                first.points = first.points + f.points(played1, trump)

                self.state[first.name] = f.card_to_number(played1, self.trump, None)  # first.points
                self.history.append({first.name: first.action})
                self.state[second.name] = f.card_to_number(played2, self.trump, played1)  # second.points
                # self.history.append({second.name: second.action})
                new_state = self.state.copy()
                self.history.append(new_state)
                first.reward = 1
                second.reward = -1

                self.dealer = first
                # logging.debug(f"{first.name} winns this round")
            else:
                # second player wins
                second.points = second.points + f.points(played2, trump)

                self.state[first.name] = f.card_to_number(played1, self.trump, None)  # first.points
                self.history.append({first.name: first.action})
                self.state[second.name] = f.card_to_number(played2, self.trump, played1)  # second.points
                # self.history.append({second.name: second.action})
                new_state = self.state.copy()
                self.history.append(new_state)
                first.reward = -1
                second.reward = 1

                self.dealer = second
                # logging.debug(f"{second.name} winns this round")

            return self.state, self.p1.reward

    def play_card(self, player, trump, played=None):
        """

        :param player: current player object
        :param trump: trump suit
        :param played: what card was played (we have to follow suit or play trump)
        :return: chosen cart to play, player hand, player played_cards
        """
        if player.name == "player1":
            # agent = AGENT1
            agent = self.p1.action
            # print(f"Agent = Player 1: {agent}")
        else:
            agent = self.p2.action
            # print(f"Agent = Player 2: {agent}")

        # if nothing is played, yet we are free to choose whatever we want
        if played is None:
            player.playable_cards = player.hand
        else:
            # figure out what cards can be played
            player.playable_cards = self.playable_cards(player.hand, trump, played)

        # if only one card is left we have to play it
        if len(player.playable_cards) == 1:
            play = player.playable_cards[0]
        else:
            # AI method hook
            ai = ai_f.AI(agent, player, trump, played, Card)
            play = ai.run()
            # print(play)
            player.hand.remove(play)
            player.played_cards.append(play)
        return play, player.hand, player.played_cards

    def playable_cards(self, hand, trump, played):
        """
        Determine playable cards in players hand

        :param hand: players hand of cards
        :param trump: trump suit
        :param played: played card if we are the second and have to follow suit
        :return:
        """
        playable_cards = []
        for card in hand:
            if card.suit == played.suit:
                playable_cards.append(card)
            elif card.suit == trump.suit:
                if card.value > trump.value:
                    playable_cards.append(card)
        if not playable_cards:
            # if there are no cards to play, we can play what we want
            playable_cards = hand
        return playable_cards


def main():
    """
    Tests for the file, does it run?
    """

    logging.info("good look")
    game = Handjass()
    for i in range(0, 1):
        g1 = game.game("state", "random", "random")
        hand = game.p1.hand
        print(i, g1, str(hand))


if __name__ == '__main__':
    main()
