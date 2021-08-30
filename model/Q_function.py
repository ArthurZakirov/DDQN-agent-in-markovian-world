import sys
import torch
import torch.nn as nn
sys.path.append('..')
from utils import batch_to_one_hot

class Q(nn.Module):
    def __init__(self, d_x, S, A):
        super().__init__()
        self.S = S
        self.A = A
        self.MLP = nn.Sequential(
            nn.Linear(d_x, A)
        )

    def forward(self, s, a=None):
        """
        Arguments
        ---------
        s : torch.tensor[bs]
        a : torch.tensor[bs]
        max : bool

        Returns
        -------
        Q : torch.tensor[bs, A]
        or
        Q_a : torch.tensor[bs]
        or
        Q_max : torch.tensor[bs], a_max
        """
        x = batch_to_one_hot(s, self.S)
        Q = self.MLP(x)

        if a != None:
            Q_a = Q[range(len(a)), a]
            return Q_a
        return Q

    def greedy(self, s, eps=0.):
        """
        Arguments
        ---------
        s : torch.tensor[1]
        eps : float

        Returns
        -------
        Q_max : torch.tensor[1]
        pi_s : torch.tensor[A]
        """
        Q = self(s)
        a_max = Q.argmax(-1)
        Q_max = Q[range(len(a_max)), a_max]
        pi_s = torch.ones(self.A) * (eps / (self.A - 1))
        pi_s[a_max] = 1 - eps
        return Q_max, a_max, pi_s