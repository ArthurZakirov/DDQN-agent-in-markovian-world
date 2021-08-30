from .Q_function import Q
import torch
import torch.nn as nn
import torch.distributions as td


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = 0

    def place(self, world):
        """place the agent into the world and initialize Q function and state

        Arguments
        ---------
        world : World

        Returns
        -------
        -
        """
        self.world = world
        self.Q = Q(d_x=world.d_x, S=world.S, A=world.A)
        self.Q_target = Q(d_x=world.d_x, S=world.S, A=world.A)
        self.s = world.initial_state
        self.pi = td.Categorical(probs=nn.Softmax(-1)(torch.randn((world.S, world.A))))

    def take_action(self):
        """
        Returns
        -------
        list(s,a,r,s)
        s : torch.tensor[1]
        a : torch.tensor[1]
        r : torch.tensor[1]
        s_next : torch.tensor[1]
        """
        e = list()
        a = self.pi.sample()[self.s]
        r, s_next = self.world.step(self.s, a)

        e.append(self.s)
        e.append(a)
        e.append(r)
        e.append(s_next)

        self.s = s_next
        return e

    def take_episode(self, T):
        """Sample s,a,r for T timesteps

        Arguments
        ---------
        T : int

        Returns
        -------
        s : torch.tensor[T,   1]
        a : torch.tensor[T-1, 1]
        r : torch.tensor[T-1, 1]
        """
        s = list()
        a = list()
        r = list()
        for t in range(T):
            s_t, a_t, r_t, s_t_next = self.take_action()
            s.append(s_t)
            a.append(a_t)
            r.append(r_t)
        s.append(s_t_next)
        return (torch.stack(s),
                torch.stack(a),
                torch.stack(r))

    def freeze_weights(self, model):
        for w in model.parameters():
            w.requires_grad = False

    def unfreeze_weights(self, model):
        for w in model.parameters():
            w.requires_grad = True

    def update_Q_target(self):
        w_target = list(self.Q_target.parameters())
        for i, w in enumerate(self.Q.parameters()):
            w_target[i].copy_(w)

    def exploration_rate(self, t):
        return (1 / (t + 1)) ** 0.5

    def learn_step(self, t, e):
        s, a, r, s_next = e
        # greedy
        _, a_max, pi_s_next = self.Q.greedy(s_next,
                                            eps=self.exploration_rate(t))

        # update policy
        self.pi.probs[s_next] = pi_s_next

        # loss
        Y = r + self.world.gamma * self.Q_target(s, a_max)
        Q_sa = self.Q(s, a)
        loss = (Y - Q_sa) ** 2
        return loss