import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Flexible MLP Block ----
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, norm=True, act='leakyrelu'):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if norm:
            layers.append(nn.LayerNorm(out_dim))
        if act == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))
        elif act == 'silu':
            layers.append(nn.SiLU())
        else:
            layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

# ---- Policy Network (Actor) ----
class MAPPOPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(MLPBlock(in_dim, hidden_dim, dropout=dropout, norm=True, act='leakyrelu'))
            in_dim = hidden_dim
        # Optional: residual connection
        self.mlp = nn.Sequential(*layers)
        self.out_head = nn.Linear(hidden_dim, n_actions)
    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        Return: logits [batch_size, n_actions]
        """
        x = self.mlp(obs)
        out = self.out_head(x)
        return out

# ---- Centralized Critic Network (Critic) ----
class MAPPOCritic(nn.Module):
    def __init__(self, global_obs_dim, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = global_obs_dim
        for _ in range(n_layers):
            layers.append(MLPBlock(in_dim, hidden_dim, dropout=dropout, norm=True, act='silu'))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.out_head = nn.Linear(hidden_dim, 1)
    def forward(self, global_obs):
        """
        global_obs: [batch_size, global_obs_dim]
        Return: state value [batch_size, 1]
        """
        x = self.mlp(global_obs)
        out = self.out_head(x)
        return out

# ---- Optional: LSTM policy for partial obs ----
class MAPPOPolicyRNN(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=512, n_layers=16):
        super().__init__()
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, n_actions)
    def forward(self, obs, hidden=None):
        # obs: [batch, seq_len, obs_dim]
        x = F.relu(self.fc(obs))
        x, hidden = self.lstm(x, hidden)
        out = self.out(x)
        return out, hidden

# ---- Pro weight initialization ----
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)

# ---- Example usage ----
if __name__ == "__main__":
    obs_dim = 100
    n_actions = 15
    n_agents = 5

    # Actor
    policy = MAPPOPolicy(obs_dim, n_actions)
    policy.apply(init_weights)

    # Critic
    critic = MAPPOCritic(obs_dim * n_agents)
    critic.apply(init_weights)

    obs_sample = torch.randn(4, obs_dim)
    global_obs_sample = torch.randn(4, obs_dim*n_agents)
    print("Policy logits:", policy(obs_sample).shape)
    print("Critic value:", critic(global_obs_sample).shape)
