


import torch
import torch.nn as nn


# class ConditionTokens(nn.Module):
#     def __init__(self, num_conditions, embed_dim=512, hidden_dim=[128, 256],  use_self_attention=False, n_heads=8, n_layers=1, dropout=0.1, use_gelu=False):
#         """
#         Projects each condition into a latent space and optionally applies a Transformer Encoder
#         to enable self-attention among tokens.

#         Parameters:
#           - num_conditions: Total number of conditions to consider.
#           - embed_dim: Dimension of the resulting embedding for each token.
#           - hidden_dim: Dimension of the hidden layers in each condition's projection.
#           - use_self_attention: Boolean flag to apply self-attention among tokens.
#           - n_heads: Number of attention heads in the Transformer Encoder.
#           - n_layers: Number of layers in the Transformer Encoder.
#         """
#         super(ConditionTokens, self).__init__()
#         self.num_conditions = num_conditions
#         self.embed_dim = embed_dim
#         self.use_self_attention = use_self_attention
#         self.use_gelu = use_gelu

#         hidden_dim = [hidden_dim] if isinstance(hidden_dim, int) else hidden_dim
#         self.projections = nn.ModuleList([
#             self.make_mlp(1, hidden_dim, embed_dim, dropout=dropout)
#             for _ in range(num_conditions)
#         ])

#         # Transformer Encoder for self-attention among tokens
#         if self.use_self_attention:
#             encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
#             self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

#     def make_mlp(self, in_dim, hidden_dim, out_dim, dropout=0.1):
#         layers = []
#         for hd in hidden_dim:
#             layers.append(nn.Linear(in_dim, hd))
#             if self.use_gelu:
#                 layers.append(nn.GELU())
#             else:
#                 layers.append(nn.ReLU())
#             if dropout > 0:
#                 # print("Using dropout in condition projection MLP")
#                 layers.append(nn.Dropout(dropout))
#             in_dim = hd
#         layers.append(nn.Linear(in_dim, out_dim))
#         layers.append(nn.LayerNorm(out_dim))
#         return nn.Sequential(*layers)

#     def forward(self, conditions):
#         """
#         conditions: Tensor of shape (batch_size, num_conditions) where each column corresponds to a specific condition.
#         """
#         tokens = torch.stack([
#             self.projections[i](conditions[:, i:i+1,0]) for i in range(self.num_conditions)
#         ], dim=1)  # (batch_size, num_conditions, embed_dim)

#         # Optionally apply self-attention among tokens
#         if self.use_self_attention:
#             tokens = tokens.transpose(0, 1)  # (num_conditions, batch_size, embed_dim)
#             tokens = self.transformer_encoder(tokens)
#             tokens = tokens.transpose(0, 1)  # (batch_size, num_conditions, embed_dim)

#         return tokens
    




class ConditionEmbedding(nn.Module):
    def __init__(
        self,
        num_conditions=3,          # [T1, T2, FLAIR]
        embed_dim=512,             # output embedding dim
        proj_hidden_dim=256,
        use_gelu=True,
        dropout=0.1
    ):
        super().__init__()

        self.num_conditions = num_conditions
        self.embed_dim = embed_dim

        # 1. Direct embedding lookup (best for one-hot categorical input)
        self.embedding = nn.Embedding(num_conditions, embed_dim)

        # 2. Optional projection to refine embedding for cross-attention
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_hidden_dim),
            nn.GELU() if use_gelu else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, conditions):
        """
        conditions: (B, 1) index tensor where each value is in [0, num_conditions-1]

        Example:
            If num_conditions=3 and we have a batch of 4 samples with conditions [0, 1, 2, 0],
            the input tensor would be:
                conditions = torch.tensor([[0], [1], [2], [0]])  # shape (4, 1)
    
            The output will be an embedding of shape (4, 1, embed_dim) ready for cross-attention.
        """

        # conditions = conditions.view(-1).long()
        # Embedding lookup: (B, embed_dim)
        # verify if conditions is of shape (B,) or (B, 1)
        if conditions.dim() == 2 and conditions.size(1) == 1:
            conditions = conditions.squeeze(1)  # (B,)
        elif conditions.dim() == 1:
            pass  # already in (B,)
        else:
            raise ValueError(f"Expected conditions to be of shape (B, 1) or (B,), but got {conditions.shape}")
        emb = self.embedding(conditions)

        # Optional refinement
        emb = self.proj(emb)

        # Return as a "token sequence" for cross-attention
        # (B, 1, embed_dim)
        return emb.unsqueeze(1)