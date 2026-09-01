import torch
import torch.nn as nn
from typing import Optional


class WanVideoActionEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int = 14,
        dim: int = 1536,
        num_action_per_chunk: Optional[int] = None,
        in_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
        ti2v2: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.dim = dim
        self.num_action_per_chunk = num_action_per_chunk
        self.ti2v2 = bool(ti2v2)

        self.action_embedding = None
        self.action_mlp1 = None
        self.action_mlp2 = None

        if self.ti2v2:
            self.action_mlp1 = nn.Sequential(
                nn.Linear(action_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            self.action_mlp2 = nn.Sequential(
                nn.Linear(action_dim * 4, 4 * dim),
                nn.SiLU(),
                nn.Linear(4 * dim, dim),
            )
        else:
            if in_features is None:
                in_features = (
                    action_dim
                    if num_action_per_chunk is None
                    else action_dim * num_action_per_chunk
                )
            if hidden_features is None:
                hidden_features = dim if num_action_per_chunk is None else dim * 4
            self.action_embedding = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(approximate='tanh'),
                nn.Linear(hidden_features, dim),
            )

    def forward(self, action):
        if self.action_embedding is None:
            raise RuntimeError(
                "Legacy action embedding is not available in TI2V2 mode."
            )
        return self.action_embedding(action)

    def encode_ti2v2(self, action: torch.Tensor):
        """Wan2.2 TI2V action encoding used by adaln mode."""
        if self.action_mlp1 is None or self.action_mlp2 is None:
            raise RuntimeError("TI2V action encoding is only available in adaln mode.")
        action_context_emb = self.action_mlp1(action)
        grouped_action = torch.cat([action[:, 0:1].repeat(1, 3, 1), action], dim=1)
        grouped_action = grouped_action.reshape(
            action.shape[0],
            (action.shape[1] + 3) // 4,
            action.shape[2] * 4,
        )
        action_mod_emb = self.action_mlp2(grouped_action)
        return action_context_emb, action_mod_emb

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        inactive_prefixes = (
            ("action_embedding.",)
            if self.action_embedding is None
            else ("action_mlp1.", "action_mlp2.")
        )
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(inactive_prefixes)
        }
        incompatible_keys = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        missing_keys = list(incompatible_keys.missing_keys)
        if strict and (missing_keys or incompatible_keys.unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict for WanVideoActionEncoder: "
                f"missing_keys={missing_keys}, "
                f"unexpected_keys={list(incompatible_keys.unexpected_keys)}"
            )
        return incompatible_keys
