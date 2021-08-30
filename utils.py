import torch
import os

def ensure_dir(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

def batch_to_one_hot(s, S):
    """
    Arguments
    ---------
    s : torch.tensor [bs, 1]
    S : int

    Returns
    -------
    s_OH : torch.tensor [bs, S]
    """
    s_OH = torch.zeros((len(s), S))
    s_OH[range(len(s)), s] = 1
    return s_OH


def reverse_one_hot(x):
    """
    Arguments
    ---------
    x : torch.tensor[bs, d_x]

    Returns
    x_int : torch.tensor[bs]
    """
    x_int = x.argmax(-1)
    return x_int


def t_every(s, s_t):
    """
    Arguments
    ---------
    s : torch.tensor [T]
    s_t : int

    Returns
    -------
    t_every : list(int, int, ...)
    """
    return [t.item() for t in np.argwhere(s[:-1] == s_t).squeeze(0)]


def t_first(s, s_t):
    """
    Arguments
    ---------
    s : torch.tensor [T]
    s_t : int

    Returns
    -------
    t_first : list(int)
    """
    return [t_every(s, s_t)[0]]