"""Model architectures for GC-GISMo (v1).

Components:
- WeightedGINConv: GIN layer that supports edge weights (matches GISMo paper Eq.).
- IngredientEncoder: Embedding + N x WeightedGINConv.
- SubstitutionDecoder: 3-layer MLP, optionally takes goal vector g.
- GISMo: top-level module. `use_health_goal=True` → MVP, else Baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing
except ImportError as e:
    raise ImportError(
        "torch_geometric is required. Install with `pip install torch_geometric`."
    ) from e


class WeightedGINConv(MessagePassing):
    """GIN layer with edge weights, matching GISMo paper:

        h_v^(l) = f^(l) ( (1 + eps^(l)) h_v^(l-1) + sum_{u in N(v)} (e_vu * h_u^(l-1)) )

    The MLP f^(l) is a 2-layer Linear → ReLU → Linear.
    """

    def __init__(self, in_dim, out_dim, eps_init=0.0):
        super().__init__(aggr="add")
        self.eps = nn.Parameter(torch.tensor(eps_init, dtype=torch.float))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_weight=None):
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = (1.0 + self.eps) * x + agg
        return self.mlp(out)

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class IngredientEncoder(nn.Module):
    """Learnable ingredient embedding -> N x WeightedGINConv.

    `num_nodes` is the TOTAL number of nodes in the graph
    (ingredients + flavor compounds + drug compounds for FlavorGraph).
    All node types share the same embedding table — F and D nodes
    get random-init learnable embeddings just like in GISMo paper.
    """

    def __init__(self, num_nodes, embed_dim=300, hidden_dim=300,
                 num_layers=2, dropout=0.25):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.gins = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(num_layers):
            self.gins.append(WeightedGINConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.dropout_p = dropout

    def forward(self, edge_index, edge_weight=None):
        # `embedding.weight` IS the full-table lookup — using it directly
        # avoids a fresh torch.arange + index_select on every forward.
        h = self.embedding.weight
        for i, gin in enumerate(self.gins):
            h = gin(h, edge_index, edge_weight)
            if i < len(self.gins) - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h  # [num_nodes, hidden_dim]


def context_embedding(h, recipe_ing_ids, recipe_mask):
    """Mean of recipe ingredient embeddings (matches GISMo CE_I).

    Source ingredient IS included in the mean (matches GISMo: recipe
    context is the full original ingredient set).

    Args:
      h:              [num_nodes, D] node embeddings from the encoder
      recipe_ing_ids: [B, max_len] padded recipe ingredient ids
      recipe_mask:    [B, max_len] 1.0 = real ingredient, 0.0 = pad

    Returns: [B, D]
    """
    h_recipe = h[recipe_ing_ids]
    h_recipe = h_recipe * recipe_mask.unsqueeze(-1)
    denom = recipe_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return h_recipe.sum(dim=1) / denom


class SubstitutionDecoder(nn.Module):
    """3-layer MLP. Input = concat(h_s, h_v, c_r [, g])."""

    def __init__(self, embed_dim=300, goal_dim=0, hidden_dim=300, dropout=0.25):
        super().__init__()
        in_dim = embed_dim * 3 + goal_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_s, h_v, c_r, g=None):
        # h_s: [B, D], h_v: [B, K, D], c_r: [B, D], g: [B, G] or None
        B, K = h_v.shape[:2]
        h_s_e = h_s.unsqueeze(1).expand(-1, K, -1)
        c_r_e = c_r.unsqueeze(1).expand(-1, K, -1)
        if g is not None:
            g_e = g.unsqueeze(1).expand(-1, K, -1)
            x = torch.cat([h_s_e, h_v, c_r_e, g_e], dim=-1)
        else:
            x = torch.cat([h_s_e, h_v, c_r_e], dim=-1)
        return self.mlp(x).squeeze(-1)  # [B, K]


class GISMo(nn.Module):
    """Full GISMo / GC-GISMo model.

    use_health_goal=False: Baseline (vanilla GISMo).
    use_health_goal=True : MVP (decoder also takes 2-dim g vector).

    `num_nodes` here is the TOTAL number of nodes (ingredients + F + D).
    Candidate enumeration in eval is restricted to ingredient ids through
    the `ingredient_ids` tensor produced by `dataset.load_node_ids`.
    """

    def __init__(self, num_nodes, embed_dim=300, hidden_dim=300,
                 num_gin_layers=2, dropout=0.25,
                 use_health_goal=False, goal_dim=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_health_goal = use_health_goal
        self.goal_dim = goal_dim if use_health_goal else 0

        self.encoder = IngredientEncoder(
            num_nodes, embed_dim, hidden_dim, num_gin_layers, dropout
        )
        self.decoder = SubstitutionDecoder(
            hidden_dim, self.goal_dim, hidden_dim, dropout
        )

    def encode_graph(self, edge_index, edge_weight=None):
        """Forward through the GNN encoder. Returns h: [num_nodes, D]."""
        return self.encoder(edge_index, edge_weight)

    def forward(self, h, source_ids, candidate_ids,
                recipe_ing_ids, recipe_mask, g=None):
        """Score (s, v, r [, g]) tuples. Returns: scores [B, K]"""
        h_s = h[source_ids]
        h_v = h[candidate_ids]
        c_r = context_embedding(h, recipe_ing_ids, recipe_mask)
        if not self.use_health_goal:
            g = None
        return self.decoder(h_s, h_v, c_r, g)
