#!/usr/bin/env python3
# File name: value_approximation.py

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

class Value_Approximation():

    def __init__(self, lamb, n_episodes, alpha=0.01, epsilon=0.05):
        self.actions = ("take", "leave")
        self.lamb = lamb  # lambda parameter of the SARSA algorithm
        self.n_episodes = n_episodes  # number of episodes (games) to sample in order to make the agent learn

        self.features = self.create_features()  # make features
        self.theta = self.init_theta()  # init theta randomly

        self.alpha = alpha
        self.epsilon = epsilon

        # used for plot
        self.Q = self.init_to_zeros()
        # self.Q_history = {}
        # self.list_n_episodes = np.linspace(10, n_episodes-1, 30, dtype=int)

    def learn_q_value_function(self):
        """
        Update the Q function until optimal value function is reached.

        Returns
        ----------
        Q : {state: (action)}, Q value for every state-action pair
        """
        for i in range(self.n_episodes):
            self.eligibilty_traces = np.zeros(56)  # init eligibilty traces to zeros
            game = env.Handjass()  # init a game sequence
            state = game.state.copy()  # init state
            action = self.e_greedy_policy(state)  # pick a first action

            while state != "terminal":
                next_state, reward = deepcopy(game.game(state, action, c.action2))

                if next_state == "terminal":
                    next_action = None
                    delta = self.compute_delta(state, action, next_state, next_action, reward)

                else:
                    next_action = self.e_greedy_policy(next_state)
                    delta = self.compute_delta(state, action, next_state, next_action, reward)

                self.update_step(delta, state, action)

                action = next_action
                state = next_state

            # if i in self.list_n_episodes:
            # self.Q_history[i] = deepcopy(self.Q)

        return None

    def init_theta(self):
        """
        Init the weights of the Value Approximation function from a normal centered reduced gaussian distribution

        Returns
        ----------
        theta : theta, a dictionnary of states as keys and actions as value
        """

        mu, sigma = 0, 0.1  # mean and standard deviation
        theta = np.random.normal(mu, sigma, 56)
        return theta

    def init_to_zeros(self):
        """
        Init the Q function and the incremental counter N at 0 for every state-action pairs.

        Returns
        ----------
        lookup_table : {}, a dictionnary of states as keys and actions as value
        """
        # space = 26 + 1
        space = c.space + 1

        players2 = np.arange(0, space)
        players1 = np.arange(0, space)
        states = [(player2, player1) for player1 in players1 for player2 in players2]
        lookup_table = {}
        for state in states:
            lookup_table[state] = {"take": 0, "leave": 0}

        return lookup_table

    def create_features(self):
        """
        Update the Q value towards the error term and eligibility traces .


        Returns
        ----------
        features : list of tuples, each tuple is a combination of features
        """

        player2 = [(42, 71), (4, 7), (7, 10), (10, 15)]
        player1 = [(312, 332), (4, 9), (7, 12), (10, 15), (13, 18), (20, 27), (26, 27)]  # expand here? -----------------
        actions = ["take", "leave"]
        features = []
        for d in player2:
            for p in player1:
                for a in actions:
                    features.append((d, p, a))
        return features

    def update_step(self, delta, state, action):
        """
        Update the weights of the linear function towards the optimality.

        Parameters
        ----------
        delta : state, the current score
        """
        lookup_state = (state["player2"], state["player1"])
        index_associated_features = []
        for i, feature in enumerate(self.features):
            if (feature[0][0] <= lookup_state[0] <= feature[0][1]) and (feature[1][0] <= lookup_state[1] <= feature[1][1]) and (action == feature[2]):
                index_associated_features.append(i)

        for i in index_associated_features:
            self.eligibilty_traces[i] = self.lamb * self.eligibilty_traces[i] + 1
            self.theta[i] = self.theta[i] + self.alpha * delta * self.eligibilty_traces[i]

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
            delta = reward - self.phi(lookup_state, action)
        else:
            next_lookup_state = (next_state["player2"], next_state["player1"])
            delta = reward + self.phi(next_lookup_state, next_action) - self.phi(lookup_state, action)
        return delta

    def phi(self, state, action):
        """
        The linear function phi, which compute the Q value for a given state-action pair.

        Parameters
        ----------
        state : state, the current score
        action : string, the current score

        Returns
        ----------
        q_value: float, the q value associated to a given state-action pair.

        """
        index_associated_features = []
        for i, feature in enumerate(self.features):
            if (feature[0][0] <= state[0] <= feature[0][1]) and (feature[1][0] <= state[1] <= feature[1][1]) and (action == feature[2]):
                index_associated_features.append(i)
        q_value = np.sum(np.take(self.theta, index_associated_features, axis=0))
        return q_value

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
        if self.epsilon > random.uniform(0, 1):
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
        if self.phi(lookup_state, "take") == self.phi(lookup_state, "leave"):
            return self.random_policy()
        else:
            if self.phi(lookup_state, "take") > self.phi(lookup_state, "leave"):
                return "take"
            else:
                return "leave"


def main():
    va = Value_Approximation(lamb=0.9, n_episodes=100)
    v = va.learn_q_value_function()
    print(v)


if __name__ == '__main__':
    main()
