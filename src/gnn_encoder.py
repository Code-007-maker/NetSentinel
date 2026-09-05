import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Add `src` rows into an output of length `dim_size` at `index`."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class GNNEncoder(nn.Module):
    """
    GraphSAGE spatial encoder that compresses a network graph G(t) into z(t).

    The 11 flow-level edge features are projected and aggregated onto incident
    nodes before SAGEConv, so gradients/perturbations on edge_attr affect z(t).
    """

    def __init__(
        self,
        node_in_dim: int = 2,
        edge_in_dim: int = 11,
        hidden_dim: int = 128,
        out_dim: int = 128,
    ):
        super(GNNEncoder, self).__init__()
        self.edge_encoder = nn.Linear(edge_in_dim, node_in_dim)
        self.conv1 = SAGEConv(node_in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        # Stabilise the scale arriving from variable-size graphs without
        # projecting every graph onto a unit sphere.  LayerNorm is applied to
        # the pooled feature vector; the latent magnitude remains learnable.
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None, return_components: bool = False):
        """
        Args:
            x: Node features (N, node_in_dim)
            edge_index: Graph connectivity (2, E)
            edge_attr: Optional edge features (E, edge_in_dim)
            batch: Batch vector for global pooling
        Returns:
            z_t: Graph-level latent state (batch_size, out_dim)
        """
        if edge_attr is not None and edge_attr.numel() > 0 and edge_index.numel() > 0:
            edge_msg = self.edge_encoder(edge_attr)
            x = x + _scatter_add(edge_msg, edge_index[0], x.size(0))
            x = x + _scatter_add(edge_msg, edge_index[1], x.size(0))

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_graph = global_mean_pool(h, batch)
        latent_pre_norm = self.fc(self.pool_norm(h_graph))
        # Do not use L2 normalisation here: it erased magnitude information and
        # allowed the temporal transition to converge to one unit direction.
        z_t = latent_pre_norm
        if return_components:
            # Kept opt-in so the model API remains compatible while diagnostics
            # can distinguish pooling collapse from projection/normalisation.
            return z_t, h_graph, latent_pre_norm
        return z_t
