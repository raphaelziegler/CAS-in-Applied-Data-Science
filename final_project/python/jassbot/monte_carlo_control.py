#!/usr/bin/env python3
# File name: monte_carlo_control.py

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


class MCControl:

    def __init__(self, N_0, n_episodes):
        self.actions = ("take", "leave")
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
                # print(state, action)
                self.increment_counter(state, action)  # increment state-action counter
                self.update_Q(state, action, reward)  # update the Q value

        return self.Q

    def init_to_zeros(self):
        """
        Init the Q function and the incremental counter N at 0 for every state-action pairs.

        Returns
        ----------
        lookup_table : {state: (action)}, a dictionnary of states as keys and actions as value
        """

        space = c.space + 1

        players2 = np.arange(0, space)
        players1 = np.arange(0, space)
        states = [(player2, player1) for player1 in players1 for player2 in players2]
        lookup_table = {}
        for state in states:
            lookup_table[state] = {"take": 0, "leave": 0}

        return lookup_table

    def play_episode(self):
        """
        Run a complete (from the initial state to the terminal state) Easy21 game sequence given a policy.

        Returns
        ----------
        episode : [(state, action, reward)], a list of (statec, action, reward)
        """
        game = env.Handjass()  # init a game sequence
        state = game.state.copy()  # init state
        episode = []  # list of the steps of the game sequence
        while state != "terminal":
            # pick an action regarding the current state and policy
            if self.policy == "random":
                action = self.random_policy()
            if self.policy == "e_greedy":
                action = self.e_greedy_policy(state)
            next_state, reward = deepcopy(game.game(state, action, c.action2))  # <------------------------------ what to do with player 2?
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
        lookup_state = (state["player2"], state["player1"])

        # The learning rate, decaying regarding the number of times an action-state pair
        # has been explored. It scales the amount of modification we want to bring to
        # the Q value function.
        alpha_t = 1 / self.get_state_action_counter(state, action)

        # We adjust the Q value towards the reality (observed) minus what we estimated.
        # This term is usually descrived as the error term.
        self.Q[lookup_state][action] += alpha_t * (reward - self.Q[lookup_state][action])

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
        # print(state)
        # print(self.N[lookup_state][action])
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
        lookup_state = (state["player2"], state["player1"])
        counter = self.N[lookup_state][action]

        return counter


def main():
    mc = MCControl(N_0=100, n_episodes=10)
    m = mc.learn_q_value_function()
    print(m)


if __name__ == '__main__':
    main()
