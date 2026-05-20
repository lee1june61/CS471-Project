"""V2: GISMo with nutrient-aware encoder.

Difference from v1 (models_v1.py):
- `IngredientEncoder` additionally projects a per-node nutrient feature vector
  through `nutrient_proj` (Linear, no bias) and adds it to the learnable
  embedding before the GIN layers.
- `bias=False` so F / D / unmapped-ingredient nodes (zero nutrient row)
  get exactly zero nutrient signal — only nodes with USDA data are nudged.
- Decoder and graph propagation are unchanged from v1 MVP. Use with goal
  vector g (v2 is always MVP-style, no baseline mode).

Extra params vs v1 MVP: nutrient_dim * embed_dim
    e.g. 7 * 300 = 2,100 — negligible compared to the 3.35M total.
"""

import torch.nn as nn
import torch.nn.functional as F

from models_v1 import WeightedGINConv, SubstitutionDecoder, context_embedding


class IngredientEncoder(nn.Module):
    """Embedding + nutrient injection + N x WeightedGINConv.

    `num_nodes` is the TOTAL number of graph nodes. `nutrient_features`
    is shape [num_nodes, nutrient_dim] — rows for F / D / unmapped
    ingredient nodes must be zero so they contribute nothing through
    the (bias-free) projection.
    """

    def __init__(self, num_nodes, nutrient_dim, embed_dim=300, hidden_dim=300,
                 num_layers=2, dropout=0.25):
        super().__init__()
        self.num_nodes = num_nodes
        self.nutrient_dim = nutrient_dim
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.nutrient_proj = nn.Linear(nutrient_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.nutrient_proj.weight)

        self.gins = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(num_layers):
            self.gins.append(WeightedGINConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.dropout_p = dropout

    def forward(self, edge_index, nutrient_features, edge_weight=None):
        h = self.embedding.weight + self.nutrient_proj(nutrient_features)
        for i, gin in enumerate(self.gins):
            h = gin(h, edge_index, edge_weight)
            if i < len(self.gins) - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h


class GISMo(nn.Module):
    """V2 GISMo: nutrient-aware encoder + goal-conditioned decoder.

    Always uses a goal vector g (no baseline mode in v2 — for the
    no-nutrient baseline run train_v1.py with --mode baseline instead).
    """

    def __init__(self, num_nodes, nutrient_dim, embed_dim=300, hidden_dim=300,
                 num_gin_layers=2, dropout=0.25, goal_dim=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.goal_dim = goal_dim

        self.encoder = IngredientEncoder(
            num_nodes, nutrient_dim, embed_dim, hidden_dim,
            num_gin_layers, dropout,
        )
        self.decoder = SubstitutionDecoder(
            hidden_dim, goal_dim, hidden_dim, dropout,
        )

    def encode_graph(self, edge_index, edge_weight, nutrient_features):
        return self.encoder(edge_index, nutrient_features, edge_weight)

    def forward(self, h, source_ids, candidate_ids,
                recipe_ing_ids, recipe_mask, g):
        """Score (s, v, r, g) tuples. Returns scores [B, K]."""
        h_s = h[source_ids]
        h_v = h[candidate_ids]
        c_r = context_embedding(h, recipe_ing_ids, recipe_mask)
        return self.decoder(h_s, h_v, c_r, g)
