import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class StateDecoder(nn.Module):
    """Decodes latent state z(t) into various network state predictions and attack probabilities."""
    def __init__(self, z_dim: int, num_mitre_tactics: int = 14):
        super(StateDecoder, self).__init__()
        
        # Attack probability head (binary)
        self.attack_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Predict the future graph latent state z_{t+1..t+K}. This must match
        # the GNN output dimensionality so the GRU rollout is trained against
        # the actual next-window latent embedding, not an unrelated 5-feature
        # summary.
        self.net_state_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim)
        )
        
        # MITRE ATT&CK tactic head (multi-class)
        self.mitre_head = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_mitre_tactics) # Outputs logits
        )
        
    def forward_logits(self, z: torch.Tensor):
        """Return raw attack logits; use BCEWithLogitsLoss during training."""
        attack_logits = self.attack_head(z)
        net_state = self.net_state_head(z)
        mitre_logits = self.mitre_head(z)
        return attack_logits, net_state, mitre_logits

    def forward(self, z: torch.Tensor):
        attack_logits, net_state, mitre_logits = self.forward_logits(z)
        return torch.sigmoid(attack_logits), net_state, mitre_logits

class TemporalWorldModel(nn.Module):
    """
    Temporal model utilizing a GRU and a transition MLP to perform recursive K-step rollouts.
    """
    def __init__(self, z_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2,
                 transition_mode: str = "direct", delta_scale: float = 0.5):
        super(TemporalWorldModel, self).__init__()
        
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        if transition_mode not in {"direct", "residual"}:
            raise ValueError("transition_mode must be 'direct' or 'residual'")
        self.transition_mode, self.delta_scale = transition_mode, float(delta_scale)
        
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
        # Bounded, but not unit-normalised: rollout states retain informative
        # scale and cannot collapse merely by sharing a single direction.
        self.last_transition_pre_activation = self.transition(last_hidden)
        delta = torch.tanh(self.last_transition_pre_activation)
        self.last_delta = self.delta_scale * delta if self.transition_mode == "residual" else None
        z_pred_next = z_seq[:, -1, :] + self.last_delta if self.transition_mode == "residual" else delta
        
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
        self.last_rollout_pre_activations = []
        self.last_rollout_deltas = []
        z_input = z_curr.unsqueeze(1) # (batch, 1, z_dim)
        h = h_curr
        
        for _ in range(k):
            out, h = self.rnn(z_input, h)
            last_hidden = out[:, -1, :]
            z_pre_activation = self.transition(last_hidden)
            self.last_rollout_pre_activations.append(z_pre_activation)
            bounded = torch.tanh(z_pre_activation)
            delta = self.delta_scale * bounded if self.transition_mode == "residual" else None
            self.last_rollout_deltas.append(delta)
            z_next = z_input[:, 0, :] + delta if self.transition_mode == "residual" else bounded
            
            predicted_zs.append(z_next)
            
            # Feed prediction back in
            z_input = z_next.unsqueeze(1)
            
        return predicted_zs
