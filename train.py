import os
import sys
import json
import torch
import numpy as np
import random
from argparse import ArgumentParser
from world import MDPWorld
from model import DoubleDQN, PrioritizedReplayDQN


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('Create world..')
    world = MDPWorld(args.S, args.A)
    print('World successfully created!')

    if args.model == 'DoubleDQN':
        agent = DoubleDQN()
        agent.place(world)
        agent.learn(T=args.T, K_freeze=args.K_freeze, lr=args.lr, log_dir=args.log_dir)

    if args.model == 'PrioritizedReplayDQN':
        agent = PrioritizedReplayDQN(args.N, args.alpha, args.beta, args.K, args.k)
        agent.place(world)
        agent.learn(T=args.T, K_freeze=args.K_freeze, lr=args.lr, log_dir=args.log_dir)

if __name__ == '__main__':
    parser = ArgumentParser()
    # files and paths
    parser.add_argument('--log_dir', type=str, help='Directory for tensorboards.', default='experiments')
    parser.add_argument('--model', type=str, help='Choose: PrioritizedReplayDQN / DoubleDQN', default='DoubleDQN')

    # MDP world
    parser.add_argument('--S', type=int, help='Number of states in the Markovian World.', default=6)
    parser.add_argument('--A', type=int, help='Number of actions in the Markovian World.', default=6)

    # Agent
    parser.add_argument('--T', type=int, help='Number of exploration steps.', default=15000)
    parser.add_argument('--K_freeze', type=int, help='Period of update of the 2nd Q function.', default=200)
    parser.add_argument('--lr', type=float, help='Learning Rate.', default=0.01)

    # Replay Memory
    parser.add_argument('--N', type=int, help='Size of the Memory.', default=100)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--K', type=int, help='Memory Replay Period.', default=10)
    parser.add_argument('--k', type=int, help='Memory Replay Batchsize.', default=100)

    args = parser.parse_args()
    main()

