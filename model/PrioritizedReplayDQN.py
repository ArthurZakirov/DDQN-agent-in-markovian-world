import os
import sys
from tqdm import tqdm
from collections import defaultdict
from utils import ensure_dir
import numpy as np
import torch
import torch.distributions as td
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from .Agent import Agent


class PrioritizedReplayDQN(Agent):
    def __init__(self, N, alpha, beta, K, k):
        super(PrioritizedReplayDQN, self).__init__()
        self.memory = Memory(N, alpha, beta)
        self.K = K
        self.k = k

    def init_cum_grad(self):
        return [torch.zeros_like(w) for w in self.Q.parameters()]

    def update_cum_grad(self, cum_grad, w_i):
        for i, w in enumerate(self.Q.parameters()):
            cum_grad[i] += w_i * w.grad

    def update_Q(self, cum_grad, lr):
        self.freeze_weights(self.Q)
        for i, w in enumerate(self.Q.parameters()):
            w.copy_(w - lr * cum_grad[i])
        self.unfreeze_weights(self.Q)

    def learn(self, T, K_freeze, lr, log_dir):
        """
        Arguments
        ---------
        T : exploration steps
        K_freeze : update period for Q_target
        K : memory replay period
        k : batchsize
        lr : learning rate
        """
        ensure_dir(log_dir)
        previous_experiments = [int(log_name[-1]) for log_name in os.listdir(log_dir) if 'exp' in log_name]
        if len(previous_experiments) == 0:
            experiment_num = 0
        else:
            experiment_num = np.array(previous_experiments).max().item()
        log_dir = f"{log_dir}/exp_{experiment_num + 1}"
        log_writer = SummaryWriter(log_dir=log_dir)
        optimizer = Adam(self.Q.parameters(), lr)

        pbar_explore = tqdm(range(T))
        pbar_explore.set_description('Explore World')

        self.freeze_weights(self.Q_target)
        for t in pbar_explore:
            e = self.take_action()
            p = torch.tensor([1.])
            self.memory.store(t, e, p)

            if t % self.K == 0:
                self.memory.update_dist()
                cum_grad = self.init_cum_grad()

                pbar_replay = tqdm(range(self.k), leave=False)
                pbar_replay.set_description('Replay Memory')
                for j_replay in pbar_replay:
                    e, j = self.memory.sample()
                    loss = self.learn_step(j, e)
                    log_writer.add_scalar('Train/loss', loss.detach(), int(t/self.K)*self.k + j_replay)
                    log_writer.add_scalar('Train/reward', e[2], int(t / self.K) * self.k + j_replay)
                    self.memory.update(j, p=loss.detach().sqrt())

                    loss.backward()
                    clip_grad_norm_(self.Q.parameters(), 1)
                    self.update_cum_grad(cum_grad, w_i=self.memory.w(j) / self.k)

                self.update_Q(cum_grad, lr)

            if t % K_freeze == 0:
                self.update_Q_target()

        self.pi.probs = self.pi.probs.round()
        torch.save(self, f'{log_dir}/model.pt')


class Memory():
    def __init__(self, N, alpha, beta):
        self.N = N

        self.alpha = alpha
        self.beta = beta

        self.H = defaultdict(dict)
        self.p_max = torch.tensor([1.])

        self.P = None

    @property
    def p(self):
        return torch.cat([H_t['p'] for t, H_t in self.H.items()])

    def w(self, j):
        P = self.P_dist.probs
        return (P.min() / P[j]) ** self.beta

    def store(self, t, e, p):
        self.H[t]['e'] = e
        if p > self.p_max:
            self.p_max = p
        self.H[t]['p'] = self.p_max

    def update_dist(self):
        P = self.p ** self.alpha
        P = P / P.sum()
        self.P_dist = td.Categorical(probs=P)

    def sample(self):
        j = self.P_dist.sample().item()
        return self.H[j]['e'], j

    def update(self, j, p):
        self.H[j]['p'] = p