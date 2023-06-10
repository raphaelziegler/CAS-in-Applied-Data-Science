#!/usr/bin/env python3
# File name: sarsa.py

# **********************************************************************
# *** The class was taken over and adapted where necessary from      ***
# *** Matyyas: "https://github.com/Matyyas/Easy21"                   ***
# **********************************************************************

# imports
import env
import numpy as np
from copy import deepcopy
import random
import config as c


class SARSA:

    def __init__(self, lamb, n_episodes, N_0=100):
        self.actions = ("take", "leave")
        self.lamb = lamb  # lambda parameter of the SARSA algorithm
        self.n_episodes = n_episodes  # number of episodes (games) to sample in order to make the agent learn
        self.N_0 = N_0  # constant parameter (influence the exploration/exploitation behavior when starting to learn)

        self.Q = self.init_to_zeros()  # init Q function to zeros
        self.N = self.init_to_zeros()  # init the counter traces to zeros

        # used for plot
        self.Q_history = {}
        self.list_n_episodes = np.linspace(10, n_episodes - 1, 30, dtype=int)

    def learn_q_value_function(self):
        """
        Update the Q function until optimal value function is reached.

        Returns
        ----------
        Q : {state: (action)}, Q value for every state-action pair
        """
        for i in range(self.n_episodes):
            self.eligibilty_traces = self.init_to_zeros()  # init eligibilty traces to zeros
            game = env.Handjass()  # init a game sequence
            state = game.state.copy()  # init state
            action = self.e_greedy_policy(state)  # pick a first action
            self.increment_counter(state, action)

            while state != "terminal":
                next_state, reward = deepcopy(game.game(state, action, c.action2))  # <------------------------------ what to do with player 2?

                if next_state == "terminal":
                    next_action = None
                    delta = self.compute_delta(state, action, next_state, next_action, reward)

                else:
                    next_action = self.e_greedy_policy(next_state)
                    delta = self.compute_delta(state, action, next_state, next_action, reward)
                    self.increment_counter(next_state, next_action)

                self.increment_eligibility_traces(state, action)
                self.update_step(delta)

                action = next_action
                state = next_state

            if i in self.list_n_episodes:
                self.Q_history[i] = deepcopy(self.Q)

        return self.Q

    def init_to_zeros(self):
        """
        Init the Q function and the incremental counter N at 0 for every state-action pairs.

        Returns
        ----------
        lookup_table : {state: (action)}, a dictionnary of states as keys and actions as value
        """
        # space = 26 + 1
        space = c.space + 1

        dealer_scores = np.arange(0, space)
        player_scores = np.arange(0, space)
        states = [(dealer_score, player_score) for player_score in player_scores for dealer_score in dealer_scores]
        lookup_table = {}
        for state in states:
            lookup_table[state] = {"take": 0, "leave": 0}

        return lookup_table

    def update_step(self, delta):
        """
        Update the Q value towards the error term and eligibility traces .

        Parameters
        ----------
        delta : float, the delta factor of the current state-action pair
        """
        for state in self.Q.keys():
            for action in self.actions:
                alpha = 1 / (self.get_state_action_counter(state, action) + 1)
                self.Q[state][action] += alpha * delta * self.eligibilty_traces[state][action]
                # Here is where the lambda parameter intervene. The higher, the longer the eligibility trace
                # associated to a state-action pair will remain
                self.eligibilty_traces[state][action] *= self.lamb
        return None

    def compute_delta(self, state, action, next_state, next_action, reward):
        """
        Update Q value towards the error term, it is the TD learning step.

        Parameters
        ----------
        state : state, the current state
        action : string, the current action
        reward : int, the current score
        next_state : int, the state we end after taking the action
        next_action : int, the action we take in next state following the policy (e greedy)

        Returns
        ----------
        delta : float, the TD error term
        """
        lookup_state = (state["player2"], state["player1"])
        if next_state == "terminal":
            delta = reward - self.Q[lookup_state][action]
        else:
            next_lookup_state = (next_state["player2"], next_state["player1"])
            delta = reward + self.Q[next_lookup_state][next_action] - self.Q[lookup_state][action]
        return delta

    def increment_eligibility_traces(self, state, action):
        """
        Increment N counter for every action-state pair encountered in an episode.

        Parameters
        ----------
        state : state, the current score
        action : string, the current score
        """
        lookup_state = (state["player2"], state["player1"])
        self.eligibilty_traces[lookup_state][action] += 1
        # print(f"ET counter for state: {lookup_state} and action: {action} --> ", self.N[lookup_state][action])
        return None

    def increment_counter(self, state, action):
        """
        Increment N counter for every action-state pair encountered in an episode.

        Parameters
        ----------
        state : state, the current score
        action : string, the current score
        """
        lookup_state = (state["player2"], state["player1"])
        self.N[lookup_state][action] += 1
        # print(f"Counter for state: {lookup_state} and action: {action} --> ", self.N[lookup_state][action])
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
        lookup_state = (state["player2"], state["player1"])
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
        lookup_state = (state["player2"], state["player1"])
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
        # print(state)
        # lookup_state = (state["dealer_score"], state["player_score"])
        counter = self.N[state][action]

        return counter


def main():
    sarsa = SARSA(lamb=0.9, n_episodes=1_000, N_0=100)
    s = sarsa.learn_q_value_function()
    print(s)


if __name__ == '__main__':
    main()
