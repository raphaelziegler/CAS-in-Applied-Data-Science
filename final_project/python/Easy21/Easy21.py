#!/usr/bin/env python3
# File name: Easy21.py

# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns; sns.set()

from copy import deepcopy
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


class Easy21():

    def __init__(self):
        """
        Init the first state by picking a random card for the dealer and player
        """
        dealer_score, _ = self.draw_card()
        player_score, _ = self.draw_card()
        self.state = {"dealer_score": dealer_score,
                      "player_score": player_score}  # initial state
        self.actions = ("hit", "stick")

        init_state = self.state.copy()  # game history, recording (state, reward) and action of each step
        self.history = [init_state]

    def step(self, state, action):
        """
        Compute a step in Easy21 game.

        Parameters
        ----------
        state : state, the current state
        action : string, the action to pick

        Returns
        -------
        state : state, new state reached given the picked action
        reward : int, the reward we get in this new state
        """
        self.history.append({"player": action})

        # player hits
        if action == "hit":
            value, color = self.draw_card()
            self.state['player_score'] = self.compute_new_score(value, color,
                                                                current_score=
                                                                self.state[
                                                                    'player_score'])

            new_state = self.state.copy()

            if self.goes_bust(self.state['player_score']):
                # player goes bust
                reward = -1
                state = "terminal"
                self.history.append(state)
                return state, reward

            else:
                reward = 0
                self.history.append(new_state)
                return self.state, reward

        # player sticks
        else:
            new_state = self.state.copy()
            self.history.append(new_state)

            state, reward = self.dealer_moves()
            return state, reward

    def draw_card(self):
        """
        Each draw from the deck results in a value between 1 and 10 (uniformly
        distributed) with a colour of red (probability 1/3) or black (probability 2/3)
        """
        value = random.randint(1, 10)
        color = ("red" if random.uniform(0, 1) <= 1 / 3 else "black")
        return value, color

    def goes_bust(self, score):
        """
        Tells if the player/dealer goes bust

        Parameters
        ----------
        score : int, the current score

        Returns
        -------
        bool : either goes bust
        """
        return ((score > 21) or (score < 1))

    def compute_new_score(self, value, color, current_score):
        """
        Compute the new score given the value and the color of the pulled card

        Parameters
        ----------
        value : int, card's value
        color : string, card's color
        current_score : int, the current score to update

        Returns
        -------
        new_score : integer
        """
        if color == "black":
            new_score = current_score + value
        else:
            new_score = current_score - value
        return new_score

    def dealer_moves(self):
        """
        Fixed dealer policy

        Returns
        -------
        state : state, the terminal state of the game sequence
        reward : int, the reward obtained in the terminal state of the game sequence
        """
        # dealer hits as long as his score is < 17
        while self.state['dealer_score'] < 17:
            value, color = self.draw_card()
            new_dealer_score = self.compute_new_score(value, color,
                                                      current_score=self.state[
                                                          'dealer_score'])
            self.state['dealer_score'] = new_dealer_score

            new_state = self.state.copy()
            self.history.append({"dealer": "hit"})
            self.history.append(new_state)

            if self.goes_bust(new_dealer_score):
                # dealer goes bust, player wins
                reward = 1
                state = "terminal"
                self.history.append(state)
                return state, reward

        self.history.append({"dealer": "stick"})

        player_score = self.state['player_score']
        dealer_score = self.state['dealer_score']

        # score > 17 -> dealer sticks
        state = "terminal"
        self.history.append(state)
        if dealer_score < player_score:  # player wins
            reward = 1
            return state, reward
        if dealer_score == player_score:  # draw
            reward = 0
            return state, reward
        if dealer_score > player_score:  # player loses
            reward = -1
            return state, reward


class MC_Control():

    def __init__(self, N_0, n_episodes):
        self.actions = ("hit", "stick")
        self.N_0 = N_0  # constant parameter (influence the exploration/exploitation behavior when starting to learn)
        self.n_episodes = n_episodes  # number of episodes (games) to sample in order to make the agent learn

        self.Q = self.init_to_zeros()  # init Q function to zeros
        self.N = self.init_to_zeros()  # init N to zeros
        self.policy = "random"  # arbitrarily init the MC learning with a random policy

    def learn_q_value_function(self):
        """
        Update the Q function until optimal value function is reached.

        Returns
        ----------
        Q : {state: (action)}, Q value for every state-action pair
        """
        for i in range(self.n_episodes):
            episode = self.play_episode()  # run an episode using current policy
            self.policy = "e_greedy"  # policy switch from random to epsilon greedy
            for step in episode:
                state, action, reward = step
                self.increment_counter(state,
                                       action)  # increment state-action counter
                self.update_Q(state, action, reward)  # update the Q value

        return self.Q

    def init_to_zeros(self):
        """
        Init the Q function and the incremental counter N at 0 for every state-action pairs.

        Returns
        ----------
        lookup_table : {state: (action)}, a dictionnary of states as keys and actions as value
        """
        dealer_scores = np.arange(1, 11)
        player_scores = np.arange(1, 22)
        states = [(dealer_score, player_score) for player_score in
                  player_scores for dealer_score in dealer_scores]
        lookup_table = {}
        for state in states:
            lookup_table[state] = {"hit": 0, "stick": 0}

        return lookup_table

    def play_episode(self):
        """
        Run a complete (from the initial state to the terminal state) Easy21 game sequence given a policy.

        Returns
        ----------
        episode : [(state, action, reward)], a list of (statec, action, reward)
        """
        easy21_game = Easy21()  # init a game sequence
        state = easy21_game.state.copy()  # init state
        episode = []  # list of the steps of the game sequence
        while state != "terminal":
            # pick an action regarding the current state and policy
            if self.policy == "random":
                action = self.random_policy()
            if self.policy == "e_greedy":
                action = self.e_greedy_policy(state)
            next_state, reward = deepcopy(easy21_game.step(state, action))
            step = (state, action, reward)
            state = next_state
            episode.append(step)

        return episode

    def update_Q(self, state, action, reward):
        """
        Update Q value towards the error term.

        Parameters
        ----------
        state : state, the current score
        action : string, the current score
        reward : int, the current score
        """
        lookup_state = (state["dealer_score"], state["player_score"])

        # The learning rate, decaying regarding the number of times an action-state pair
        # has been explored. It scale the amount of modification we want to bring to
        # the Q value function.
        alpha_t = 1 / self.get_state_action_counter(state, action)

        # We adjust the Q value towards the reality (observed) minus what we estimated.
        # This term is usually descrived as the error term.
        self.Q[lookup_state][action] += alpha_t * (
                    reward - self.Q[lookup_state][action])

        return None

    def increment_counter(self, state, action):
        """
        Increment N counter for every action-state pair encountered in an episode.

        Parameters
        ----------
        state : state, the current score
        action : string, the current score
        """
        lookup_state = (state["dealer_score"], state["player_score"])
        # print(state)
        self.N[lookup_state][action] += 1
        return None

    def random_policy(self):
        """
        Return an action follwing a random policy (state free).

        Returns
        ----------
        action : string, random action
        """
        action = random.choice(self.actions)

        return action

    def e_greedy_policy(self, state):
        """
        Return an action given an epsilon greedy policy (state based).

        Parameters
        ----------
        state : state, state where we pick the action

        Returns
        ----------
        action : string, action from epsilon greedy policy
        """
        e = self.N_0 / (self.N_0 + self.get_state_counter(state))
        if e > random.uniform(0, 1):
            action = random.choice(self.actions)
        else:
            action = self.get_action_w_max_value(state)

        return action

    def get_action_w_max_value(self, state):
        """
        Return the action with the max Q value at a given state.

        Parameters
        ----------
        state : state, state

        Returns
        ----------
        action : string, action from epsilon greedy policy
        """
        lookup_state = (state["dealer_score"], state["player_score"])
        list_values = list(self.Q[lookup_state].values())
        if list_values[0] == list_values[1]:
            return self.random_policy()
        else:
            action = max(self.Q[lookup_state], key=self.Q[lookup_state].get)
            return action

    def get_state_counter(self, state):
        """
        Return the counter for a given state.

        Parameters
        ----------
        state : state, state

        Returns
        ----------
        counter : int, the number of times a state as been explored
        """
        lookup_state = (state["dealer_score"], state["player_score"])
        counter = np.sum(list(self.N[lookup_state].values()))

        return counter

    def get_state_action_counter(self, state, action):
        """
        Return the counter for a given action-state pair.

        Parameters
        ----------
        state : state
        action : string

        Returns
        ----------
        counter : int, the number of times an action-state pair as been explored
        """
        lookup_state = (state["dealer_score"], state["player_score"])
        counter = self.N[lookup_state][action]

        return counter


def plot_Q(n_episodes):
    mc = MC_Control(N_0=100, n_episodes=n_episodes)
    mc.learn_q_value_function()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection='3d')

    # Make data.
    dealer_showing = np.arange(1, 11)
    player_score = np.arange(1, 22)
    dealer_showing, player_score = np.meshgrid(dealer_showing, player_score)

    max_Q = np.ndarray(shape=(21, 10))
    for state in mc.Q:
        max_Q[state[1] - 1][state[0] - 1] = max(mc.Q[state].values())

    # Plot the surface.
    surf = ax.plot_surface(dealer_showing, player_score, max_Q,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize plot
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel('Dealer showing', fontsize=12)
    plt.ylabel('Player score', fontsize=12)
    plt.title('Optimal Q value function', fontsize=16)

    plt.xticks(np.arange(1, 11))
    plt.yticks(np.arange(1, 22))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_opt_policy(n_episodes):
    mc = MC_Control(N_0=100, n_episodes=n_episodes)
    mc.learn_q_value_function()
    df = pd.DataFrame(columns=["dealer_showing", "player_score", "best_action"])
    states = list(mc.Q.keys())
    for i in range(len(states)):
        best_action = max(mc.Q[states[i]], key=mc.Q[states[i]].get)
        df.loc[i] = (states[i][0], states[i][1], best_action)

    df_pivot = df.pivot("player_score", "dealer_showing", "best_action")
    return df_pivot


def main():
    # Explore a game of Easy21 to check if the environment is coherent with Easy21 rules
    # easy21_game = Easy21()
    # state_0 = easy21_game.state
    # state_1 = easy21_game.step(state_0, "hit")[0]
    # state_2 = easy21_game.step(state=state_1, action="stick")
    # print(easy21_game.history)

    # mc = MC_Control(N_0=100, n_episodes=100)
    # mc.learn_q_value_function()

    plot_opt_policy(n_episodes=100)

if __name__ == '__main__':
    main()
