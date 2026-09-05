import torch
import torch.nn as nn
from typing import Tuple, List

class StateDecoder(nn.Module):
    """Decodes latent state z(t) into various network state predictions and attack probabilities."""
    def __init__(self, z_dim: int, num_mitre_tactics: int = 14):
        super(StateDecoder, self).__init__()
        
        # Attack probability head (binary)
        self.attack_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Next network state features (e.g. flow count, avg duration, total bytes)
        # Let's say we predict 5 macroscopic network features
        self.net_state_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # MITRE ATT&CK tactic head (multi-class)
        self.mitre_head = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_mitre_tactics) # Outputs logits
        )
        
    def forward(self, z: torch.Tensor):
        attack_prob = self.attack_head(z)
        net_state = self.net_state_head(z)
        mitre_logits = self.mitre_head(z)
        return attack_prob, net_state, mitre_logits

class TemporalWorldModel(nn.Module):
    """
    Temporal model utilizing a GRU and a transition MLP to perform recursive K-step rollouts.
    """
    def __init__(self, z_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super(TemporalWorldModel, self).__init__()
        
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        
        # RNN to maintain temporal hidden state
        self.rnn = nn.GRU(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Transition model to predict next z given hidden state
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        
    def forward(self, z_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a sequence of historical z to update hidden state.
        Args:
            z_seq: (batch_size, seq_len, z_dim)
        Returns:
            z_pred_next: (batch_size, z_dim) predicted next state for t+1
            h_n: hidden state to be used for rollout
        """
        # output: (batch, seq_len, hidden_dim), h_n: (num_layers, batch, hidden_dim)
        output, h_n = self.rnn(z_seq)
        
        # We take the output of the last timestep to predict the next z
        last_hidden = output[:, -1, :]
        z_pred_next = self.transition(last_hidden)
        
        return z_pred_next, h_n

    def rollout(self, z_curr: torch.Tensor, h_curr: torch.Tensor, k: int) -> List[torch.Tensor]:
        """
        Genuine recursive K-step rollout:
        z(t) -> predict z(t+1) -> use z(t+1) to predict z(t+2) ...
        
        Args:
            z_curr: (batch_size, z_dim) Current state z(t) or predicted z(t)
            h_curr: (num_layers, batch, hidden_dim) Current RNN hidden state
            k: number of steps to roll out
            
        Returns:
            List of predicted z vectors: [z(t+1), z(t+2), ..., z(t+K)]
        """
        predicted_zs = []
        z_input = z_curr.unsqueeze(1) # (batch, 1, z_dim)
        h = h_curr
        
        for _ in range(k):
            out, h = self.rnn(z_input, h)
            last_hidden = out[:, -1, :]
            z_next = self.transition(last_hidden)
            
            predicted_zs.append(z_next)
            
            # Feed prediction back in
            z_input = z_next.unsqueeze(1)
            
        return predicted_zs
