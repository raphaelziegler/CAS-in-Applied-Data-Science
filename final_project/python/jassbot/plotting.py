#!/usr/bin/env python3
# File name: plotting.py
import logging
import time

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from monte_carlo_control import MCControl
from sarsa import SARSA
from value_approximation import Value_Approximation

from multiprocessing import Pool

import cProfile
import pstats

import locale
locale.setlocale(locale.LC_NUMERIC, 'de_CH.utf8')

import config as c


def mcc_plot_Q(n_episodes):
    start = time.time()
    mc = MCControl(N_0=100, n_episodes=n_episodes)
    p = mc.learn_q_value_function()
    print(f"Runtime: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}\n\n")

    fig = plt.figure(figsize=(20, 10))
    # ax = fig.gca(projection='3d')  # not working
    ax = fig.add_subplot(projection='3d')

    # Make data.
    dealer_showing = np.arange(0, c.space)
    player_score = np.arange(0, c.space)
    dealer_showing, player_score = np.meshgrid(dealer_showing, player_score)

    max_Q = np.ndarray(shape=(c.space, c.space))
    for state in mc.Q:
        max_Q[state[1] - 1][state[0] - 1] = max(mc.Q[state].values())

    # Plot the surface.
    surf = ax.plot_surface(dealer_showing, player_score, max_Q, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # save max Q values to file for use in the op methode in the ai_functionality module
    with open(f"data/{c.metric}_mcc_max_Q_array_{n_episodes}_{c.num_to_card}.npy", "wb") as file:
        np.save(file, max_Q)

    # Customize plot
    # ax.set_zlim(-1.01, 1.01)
    ax.set_zlim(max_Q.min() - 0.01, max_Q.max() + 0.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel('Player 2', fontsize=12)
    plt.ylabel('Player 1', fontsize=12)
    plt.title(f"MCC: Optimal Q value function\nMetric: {c.metric}\nNumber of episodes {n_episodes:n}\nPlayer 2 choice: {c.action2}", fontsize=16)

    plt.xticks(np.arange(0, c.space, c.ax_display))
    plt.yticks(np.arange(0, c.space, c.ax_display))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def sarsa_plot_Q(n_episodes):
    start = time.time()
    sarsa = SARSA(0.9, n_episodes=n_episodes)
    sarsa.learn_q_value_function()
    print(f"Runtime: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}\n\n")

    fig = plt.figure(figsize=(20, 10))
    # ax = fig.gca(projection='3d')  # not working
    ax = fig.add_subplot(projection='3d')

    # Make data.
    dealer_showing = np.arange(0, c.space)
    player_score = np.arange(0, c.space)
    dealer_showing, player_score = np.meshgrid(dealer_showing, player_score)

    max_Q = np.ndarray(shape=(c.space, c.space))
    for state in sarsa.Q:
        max_Q[state[1] - 1][state[0] - 1] = max(sarsa.Q[state].values())

    # Plot the surface.
    surf = ax.plot_surface(dealer_showing, player_score, max_Q, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # save max Q values to file for use in the op methode in the ai_functionality module
    with open(f"data/{c.metric}_sarsa_max_Q_array_{n_episodes}_{c.num_to_card}.npy", "wb") as file:
        np.save(file, max_Q)

    # Customize plot
    ax.set_zlim(max_Q.min() - 0.01, max_Q.max() + 0.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel('Player 2', fontsize=12)
    plt.ylabel('Player 1', fontsize=12)
    plt.title(f"SARSA: Optimal Q value function\nMetric: {c.metric}\nNumber of episodes {n_episodes:n}\nPlayer 2 choice: {c.action2}", fontsize=16)

    plt.xticks(np.arange(0, c.space, c.ax_display))
    plt.yticks(np.arange(0, c.space, c.ax_display))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def mcc_plot_opt_policy(n_episodes):
    mc = MCControl(N_0=100, n_episodes=n_episodes)
    mc.learn_q_value_function()
    df = pd.DataFrame(columns=["player2", "player1", "best_action"])
    states = list(mc.Q.keys())
    for i in range(len(states)):
        best_action = max(mc.Q[states[i]], key=mc.Q[states[i]].get)
        df.loc[i] = (states[i][0], states[i][1], best_action)

    # df_pivot = df.pivot("player1", "player2", "best_action")  # not working
    df_pivot = df.pivot(index="player1", columns="player2", values="best_action")
    return df_pivot


def mcc_display_plot_opt_policy(n_episodes):
    pd.set_option("expand_frame_repr", False)
    r = mcc_plot_opt_policy(n_episodes=n_episodes)
    r.to_csv(f"data/{c.metric}_mcc_optimal_policy_{n_episodes}_{c.num_to_card}.csv")
    print(r)
    pd.reset_option("expand_frame_repr")

def MSE(optimal_Q, sarsa_Q):
    actions = ["take", "leave"]
    mse = 0
    for state in optimal_Q.keys():
        for action in actions:
            mse += (sarsa_Q[state][action] - optimal_Q[state][action])**2
    mse *= (1 / len(actions)*len(optimal_Q.keys()))
    return mse


def plot_mse(optimal_Q, n_episodes, list_lambdas):
    df = pd.DataFrame(columns=["mse", "n_episodes", "lambda"])
    #list_lambdas = np.linspace(0,1,11)
    i = 0

    for lamb in list_lambdas:
        sarsa_Q = SARSA(lamb=lamb, n_episodes=n_episodes, N_0=100)
        sarsa_Q.learn_q_value_function()
        list_n_episodes = sarsa_Q.list_n_episodes

        for n_episodes in list_n_episodes:
            mse = MSE(optimal_Q, sarsa_Q.Q_history[n_episodes])
            lamb = round(lamb, 2)
            df.loc[i] = (mse, int(n_episodes), f"lambda = {lamb}")
            i += 1

    fig_dims = (12, 8)
    fig, ax = plt.subplots(figsize=fig_dims)
    title = f"Metric: {c.metric}\nMSE MCC-SARSA"
    sns.lineplot(x="n_episodes", y="mse", hue="lambda", data=df).set(title=title)
    plt.show()


def sarsa_mcc_convergence_plot_lamdas(n_episodes):
    mc = MCControl(N_0=100, n_episodes=n_episodes)
    optimal_Q = mc.learn_q_value_function()

    list_lamdas = np.linspace(0, 1, 11)
    # list_lamdas = [0, 1]  # min, max
    n_episodes = int(n_episodes / 2)
    plot_mse(optimal_Q=optimal_Q, n_episodes=n_episodes, list_lambdas=list_lamdas)


def sarsa_plot_opt_policy(lamb, n_episodes):
    sarsa_Q = SARSA(lamb=lamb, n_episodes=n_episodes, N_0=100)
    sarsa_Q.learn_q_value_function()
    df = pd.DataFrame(columns=["player2", "player1", "best_action"])
    states = list(sarsa_Q.Q.keys())
    for i in range(len(states)):
        best_action = max(sarsa_Q.Q[states[i]], key=sarsa_Q.Q[states[i]].get)
        df.loc[i] = (states[i][0], states[i][1], best_action)

    # df_pivot = df.pivot("player1", "player2", "best_action")  # not working
    df_pivot = df.pivot(index="player1", columns="player2", values="best_action")
    return df_pivot


def sarsa_display_plot_opt_policy(lamb, n_episodes):
    pd.set_option("expand_frame_repr", False)
    r = sarsa_plot_opt_policy(lamb=lamb, n_episodes=n_episodes)
    r.to_csv(f"{c.metric}_sarsa_optimal_policy_{n_episodes}_{c.num_to_card}.csv")
    print(r)
    pd.reset_option("expand_frame_repr")


def plot_va_Q(n_episodes, lamb=0.9):
    va = Value_Approximation(lamb=lamb, n_episodes=n_episodes)
    va.learn_q_value_function()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(projection='3d')

    # Make data
    player2 = np.arange(0, c.space)
    player1 = np.arange(0, c.space)
    player2, player1 = np.meshgrid(player2, player1)

    Q = {}
    for state in va.Q.keys():
        for action in va.actions:
            Q[state] = {"take": va.phi(state, "take"), "leave": va.phi(state, "leave")}

    max_Q = np.ndarray(shape=(c.space, c.space))
    for state in va.Q:
        max_Q[state[1] - 1][state[0] - 1] = max(Q[state].values())

    # Plot the surface
    surf = ax.plot_surface(player2, player1, max_Q, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize plot
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel('Player2', fontsize=12)
    plt.ylabel('Player1', fontsize=12)
    plt.title(f"VA: Optimal Q value function\nMetric: {c.metric}", fontsize=16)

    plt.xticks(np.arange(0, c.space, c.ax_display))
    plt.yticks(np.arange(0, c.space, c.ax_display))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def main(n_episodes=100, sarsa_n_episodes=100):
    # parallel plotting, threading

    pool = Pool(processes=5)
    thread1 = pool.apply_async(mcc_plot_Q, [n_episodes])
    thread2 = pool.apply_async(mcc_display_plot_opt_policy, [n_episodes])
    thread3 = pool.apply_async(sarsa_mcc_convergence_plot_lamdas, [sarsa_n_episodes])
    thread4 = pool.apply_async(sarsa_display_plot_opt_policy, [0.9, sarsa_n_episodes])
    thread5 = pool.apply_async(plot_va_Q, [n_episodes])
    thread6 = pool.apply_async(sarsa_plot_Q(sarsa_n_episodes))

    pool.close()
    pool.join()

    print(f"all processes done")


def profile():
    # profile code to finde slow functions

    with cProfile.Profile() as pr:
        mcc_plot_Q(1_000)
        # sarsa_plot_Q(10)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    # stats.dump_stats(filename="profiling.prof")


if __name__ == '__main__':
    mcc_plot_Q(1_000)
    # sarsa_plot_Q(10000)
    # mcc_display_plot_opt_policy(100)
    # sarsa_mcc_convergence_plot_lamdas(20_000)
    # sarsa_display_plot_opt_policy(0.9, 100)
    # plot_va_Q(100)

    # main(10_000, 1_000)

    # profile()
