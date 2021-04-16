import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from modules_tools import *


class NoMemory(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_dim = 0
        self._empty = nn.parameter.Parameter(torch.FloatTensor(), requires_grad=False)  # To keep track of device

    def forward(self, embed, action, reset, in_state):
        return (in_state,)

    def init_state(self, batch_size):
        return self._empty

    def loss(self, *args):
        return torch.tensor(0.0, device=self._empty.device)


class GlobalStateMem(nn.Module):

    def __init__(self, embed_dim=256, action_dim=7, mem_dim=200, stoch_dim=30, hidden_dim=200, min_std=0.1):
        super().__init__()
        self._cell = GlobalStateCell(embed_dim, action_dim, mem_dim, stoch_dim, hidden_dim, min_std)
        self.global_dim = stoch_dim

    def forward(self,
                embed,     # tensor(N, B, E)
                action,    # tensor(N, B, A)
                reset,     # tensor(N, B)
                in_state_post,  # (tensor(B, M), tensor(B, S))
                ):

        n = embed.size(0)
        states = []
        posts = []
        state = in_state_post[0]

        for i in range(n):
            state, post = self._cell(embed[i], action[i], reset[i], state)
            states.append(state)
            posts.append(post)

        sample = diag_normal(posts[-1]).rsample()
        out_state_post = (states[-1].detach(), posts[-1].detach())

        return (
            sample,                      # tensor(   B, S)
            torch.stack(states),         # tensor(N, B, M)
            torch.stack(posts),          # tensor(N, B, 2S)
            in_state_post,               # (tensor(B, M), tensor(B, 2S)) - for loss
            out_state_post,              # (tensor(B, M), tensor(B, 2S))
        )

    def init_state(self, batch_size):
        return self._cell.init_state(batch_size)

    def loss(self,
             sample, states, posts, in_state_post, out_state_post,       # forward() output
             ):
        in_post = in_state_post[1]
        priors = torch.cat([in_post.unsqueeze(0), posts[:-1]])
        loss_kl = D.kl.kl_divergence(diag_normal(posts), diag_normal(priors))  # KL between consecutive posteriors
        loss_kl = loss_kl.mean()        # (N, B) => ()
        return loss_kl


class GlobalStateCell(nn.Module):

    def __init__(self, embed_dim=256, action_dim=7, mem_dim=200, stoch_dim=30, hidden_dim=200, min_std=0.1):
        super().__init__()
        self._mem_dim = mem_dim
        self._min_std = min_std

        self._ea_mlp = nn.Sequential(nn.Linear(embed_dim + action_dim, hidden_dim),
                                     nn.ELU())

        self._gru = nn.GRUCell(hidden_dim, mem_dim)

        self._post_mlp = nn.Sequential(nn.Linear(mem_dim, hidden_dim),
                                       nn.ELU(),
                                       nn.Linear(hidden_dim, 2 * stoch_dim))

    def init_state(self, batch_size):
        device = next(self._gru.parameters()).device
        state = torch.zeros((batch_size, self._mem_dim), device=device)
        post = to_mean_std(self._post_mlp(state), self._min_std)
        return (state.detach(), post.detach())

    def forward(self,
                embed,     # tensor(B, E)
                action,    # tensor(B, A)
                reset,     # tensor(B)
                in_state,  # tensor(B, M)
                ):

        in_state = in_state * ~reset.unsqueeze(1)

        ea = self._ea_mlp(cat(embed, action))                                # (B, H)
        state = self._gru(ea, in_state)                                     # (B, M)
        post = to_mean_std(self._post_mlp(state), self._min_std)            # (B, 2*S)

        return (
            state,           # tensor(B, M)
            post,            # tensor(B, 2*S)
        )