"""V4: GISMo with decoder-level nutrient injection.

Difference from v1 MVP / v2 / v3:
- Encoder is identical to v1 (no nutrient feature addition, no hub nodes).
  The graph is the base FlavorGraph, same as the baseline.
- Decoder receives raw (log1p + z-score) nutrient features for the source
  and each candidate as extra concat inputs alongside (h_s, h_v, c_r, g).

Motivation: v2 (encoder feature) and v3 (hub-node structure) both inject
nutrient signal *before* the GIN. The 2-layer aggregation mixes each
ingredient's nutrient with its neighbors' (and in v3, with a hub that has
already averaged over many ingredients), diluting the per-ingredient
information. v4 injects after the encoder, so the score function sees raw
nutrient values directly  no GIN smoothing.

Implementation notes:
- nutrient_dim defaults to 2 (sugar_g, sodium_mg), matching the goal
  vector dimensions. This keeps a direct semantic mapping: g[k] and
  (n_s[k] - n_v[k]) refer to the same nutrient, so the decoder can
  learn the obvious "if g_sugar=1 prefer candidates with smaller n_sugar"
  rule directly through its MLP.
- Extra params vs v1 MVP: 4 input dims x 300 hidden = 1,200. Negligible
  compared to the ~3.35M total.
"""

import torch
import torch.nn as nn

from models_v1 import IngredientEncoder, context_embedding


class SubstitutionDecoderWithNutrient(nn.Module):
    """3-layer MLP. Input = concat(h_s, h_v, c_r, g, n_s, n_v).

    Identical structure to v1's SubstitutionDecoder. Only the input
    dimension grows by 2 * nutrient_dim to accommodate raw nutrient
    features for source and candidate.
    """

    def __init__(self, embed_dim=300, goal_dim=2, nutrient_dim=2,
                 hidden_dim=300, dropout=0.25):
        super().__init__()
        in_dim = embed_dim * 3 + goal_dim + nutrient_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_s, h_v, c_r, g, n_s, n_v):
        # h_s: [B, D]      h_v: [B, K, D]   c_r: [B, D]
        # g:   [B, G]      n_s: [B, n_dim]  n_v: [B, K, n_dim]
        B, K = h_v.shape[:2]
        h_s_e = h_s.unsqueeze(1).expand(-1, K, -1)
        c_r_e = c_r.unsqueeze(1).expand(-1, K, -1)
        g_e = g.unsqueeze(1).expand(-1, K, -1)
        n_s_e = n_s.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([h_s_e, h_v, c_r_e, g_e, n_s_e, n_v], dim=-1)
        return self.mlp(x).squeeze(-1)  # [B, K]


class GISMo(nn.Module):
    """V4 GISMo: v1 encoder + nutrient-aware decoder.

    Always uses a goal vector g (no baseline mode). For the no-health
    baseline, run train_v1.py --mode baseline.
    """

    def __init__(self, num_nodes, nutrient_dim=2, embed_dim=300,
                 hidden_dim=300, num_gin_layers=2, dropout=0.25, goal_dim=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_health_goal = True
        self.goal_dim = goal_dim
        self.nutrient_dim = nutrient_dim

        self.encoder = IngredientEncoder(
            num_nodes, embed_dim, hidden_dim, num_gin_layers, dropout,
        )
        self.decoder = SubstitutionDecoderWithNutrient(
            hidden_dim, goal_dim, nutrient_dim, hidden_dim, dropout,
        )

    def encode_graph(self, edge_index, edge_weight=None):
        """v1 encoder  no nutrient pass-through."""
        return self.encoder(edge_index, edge_weight)

    def forward(self, h, source_ids, candidate_ids,
                recipe_ing_ids, recipe_mask, g, nutrient_tensor):
        """Score (s, v, r, g, n_s, n_v) tuples. Returns scores [B, K].

        nutrient_tensor: [num_total_nodes, nutrient_dim], same tensor used
        for L_health — per-ingredient (sugar, sodium) in log1p + z-score
        form. Sized for num_total_nodes so raw source/candidate ids index
        directly even with sparse ingredient id space.
        """
        h_s = h[source_ids]
        h_v = h[candidate_ids]
        c_r = context_embedding(h, recipe_ing_ids, recipe_mask)
        n_s = nutrient_tensor[source_ids]
        n_v = nutrient_tensor[candidate_ids]
        return self.decoder(h_s, h_v, c_r, g, n_s, n_v)
