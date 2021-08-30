import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from utils import batch_to_one_hot


class MDP():
    """
    Attributes
    ----------
    S : int
        number of states

    d_x : int
        number of state features

    A : int
        number of actions

    P : torch.tensor [S, A, S]
        transition probabilities

    P_dist : td.Categorical
        used to sample new state when agent performs action

    R : torch.tensor [S, A]
        rewards

    gamma : float
        discount factor


    Property
    --------
    pi_opt
    V_opt
    Q_opt
    """

    def __init__(self, S, A):
        self.A = A
        self.S = S
        self.d_x = S
        self.P = nn.Softmax(-1)(torch.randn((S, A, S)))
        self.P_dist = td.Categorical(probs=self.P)

        self.R = torch.randint(0, 10, size=(S, A))

        self.gamma = 0.5

    def V_next(self, V, pi):
        """ V(s) - Value-State, k-step

        Arguments
        ---------
        pi : torch.tensor[S, A]
        V : torch.tensor[S]

        Returns
        -------
        V : torch.tensor[S]
        """
        V_next = torch.stack([self.R[s, pi[s].argmax()] \
                              + self.gamma * (self.P[s, pi[s].argmax()] * V).sum()
                              for s in range(self.S)])
        return V_next

    def Q_next(self, V, a):
        """ Q(s,a) - Action-Value-State, k-step

        Arguments
        ---------
        a : int
        V : torch.tensor[S]

        Returns
        -------
        Q_next : torch.tensor[S]
        """
        Q_next = torch.stack([
            self.R[s, a] + self.gamma * (self.P[s, a] * V).sum()
            for s in range(self.S)
        ])
        return Q_next

    def a_next_opt(self, V):
        """
        Arguments
        ---------
        V : torch.tensor[S]

        Returns
        -------
        a_max : torch.tensor[S]
        """
        Q_a = torch.stack([self.Q_next(V, a) for a in range(self.A)], dim=-1)
        a_max = Q_a.argmax(-1)
        return a_max

    def V(self, pi):
        """ V(s) - State-Value
        Arguments
        ---------
        pi : torch.tensor [S,A]

        Returns
        -------
        V : torch.tensor[S]
        """
        k = 0
        while True:
            if k == 0:
                V = torch.zeros(self.S)
            if k >= 1:
                V_prev = V
                V = self.V_next(V, pi)
                if np.linalg.norm(V - V_prev) < 0.01:
                    break
            k += 1
        return V

    def Q(self, a, pi):
        """ Q(s,a) - Action-State-Value

        Arguments
        ---------
        pi : torch.tensor [S,A]
        a : int

        Returns
        -------
        Q : np.array[S]
        """
        Q = torch.stack([self.R[s, a] \
                         + self.gamma * (self.P[s, a] * self.V(pi)).sum()
                         for s in range(self.S)])
        return Q

    @property
    def pi_opt(self):
        k = 0
        while True:
            if k == 0:
                V = torch.zeros(self.S)
                # the initial pi is not necessary for the algorithm
                # it just makes it easer to define when to stop the iteration
                pi = nn.Softmax(-1)(torch.randn((self.S, self.A)))
            if k >= 1:
                pi_prev = pi
                a_opt = self.a_next_opt(V)
                pi = batch_to_one_hot(a_opt, self.A)
                V = self.V_next(V, pi)

                if torch.equal(pi, pi_prev):
                    break
            k += 1
        return pi

    @property
    def V_opt(self):
        return self.V(self.pi_opt)

    @property
    def Q_opt(self):
        return torch.stack([self.Q(a, self.pi_opt) for a in range(self.A)], dim=1)


class MDPWorld(MDP):
    def __init__(self, S, A):
        super(MDPWorld, self).__init__(S, A)

    @property
    def initial_state(self):
        """
        Returns
        -------
        s : torch.tensor[1]
        """
        return torch.tensor([0])

    def step(self, s, a):
        """
        Arguments
        ---------
        s : torch.tensor[1]
        a : torch.tensor[1]

        Returns
        -------
        r : torch.tensor[1]
        s : torch.tensor[1]
        """
        r = self.R[s, a]
        s_next = self.P_dist.sample()[s, a]
        return r, s_next