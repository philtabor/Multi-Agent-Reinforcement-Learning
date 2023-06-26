import numpy as np
from utils import plot_learning_curve

maddpg_scores = np.load('data/maddpg_scores.npy')
maddpg_steps = np.load('data/maddpg_steps.npy')

ddpg_scores = np.load('data/ddpg_scores.npy')
ddpg_steps = np.load('data/ddpg_steps.npy')

plot_learning_curve(x=maddpg_steps,
                    scores=(maddpg_scores, ddpg_scores),
                    filename='plots/maddpg_vs_ddpg.png')
