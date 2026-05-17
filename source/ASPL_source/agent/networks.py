import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, num_layers=3):
        super(Actor, self).__init__()

        self.max_action = max_action

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))  

    
        self.layers = nn.Sequential(*layers)
        for m in self.layers:
            if isinstance(m, nn.Linear) and m.out_features == hidden_dim:
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        last = self.layers[-1]
        if isinstance(last, nn.Linear) and last.out_features == action_dim:
            nn.init.uniform_(last.weight, -1e-3, 1e-3)
            if last.bias is not None:
                nn.init.uniform_(last.bias, -1e-3, 1e-3)

    def forward(self, state):

        x = self.layers(state)
        return self.max_action * torch.tanh(x)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super(Critic, self).__init__()

        input_dim = state_dim + action_dim

        self.q_networks = nn.ModuleList([
            self._build_q_network(input_dim, hidden_dim, num_layers),
            self._build_q_network(input_dim, hidden_dim, num_layers),
        ])

    @staticmethod
    def _build_q_network(input_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers):
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))

        net = nn.Sequential(*layers)

        for m in net:
            if isinstance(m, nn.Linear) and m.out_features == hidden_dim:
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        last = net[-1]
        if isinstance(last, nn.Linear) and last.out_features == 1:
            nn.init.uniform_(last.weight, -3e-3, 3e-3)
            if last.bias is not None:
                nn.init.uniform_(last.bias, -3e-3, 3e-3)

        return net

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.q_networks[0](sa)
        q2 = self.q_networks[1](sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q_networks[0](sa)