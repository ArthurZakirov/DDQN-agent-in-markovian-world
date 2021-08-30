import os
import sys
from tqdm import tqdm
from .Agent import Agent
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
sys.path.append('..')
from utils import ensure_dir


class DoubleDQN(Agent):
    def __init__(self):
        super(DoubleDQN, self).__init__()

    def learn(self, T, K_freeze, lr, log_dir):
        """
        Arguments
        ---------
        T : exploration steps
        K_freeze : update period for Q_target
        lr : learning rate
        """

        # Training Preparation
        ensure_dir(log_dir)
        previous_experiments = [int(log_name[-1]) for log_name in os.listdir(log_dir) if 'exp' in log_name]
        if len(previous_experiments) == 0:
            experiment_num = 0
        else:
            experiment_num = np.array(previous_experiments).max().item()
        log_dir = f"{log_dir}/exp_{experiment_num+1}"
        log_writer = SummaryWriter(log_dir=log_dir)
        optimizer = Adam(self.Q.parameters(), lr)

        V_opt_sum = self.world.V_opt.sum()
        log_rewards = list()

        self.freeze_weights(self.Q_target)
        for t in tqdm(range(T)):
            # take action
            e = self.take_action()
            loss = self.learn_step(t, e)

            # update Q weights
            loss.backward()
            clip_grad_norm_(self.Q.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()


            if t % K_freeze == 0:
                self.update_Q_target()

            log_rewards.append(e[2])
            if t % 10 == 0:
                log_writer.add_scalar(f'Train/reward', np.array(log_rewards).mean(), t)
                log_rewards = list()
            log_writer.add_scalar('Train/loss', loss.detach(), t)
            V_agent_sum = self.world.V(pi=self.pi.probs.round()).sum()
            log_writer.add_scalar('Train/V_sum', V_agent_sum/V_opt_sum, t)

        self.pi.probs = self.pi.probs.round()
        torch.save(self, f'{log_dir}/model.pt')