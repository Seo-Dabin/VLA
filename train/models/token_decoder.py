"""Stage 3 Token Decoder: Transformer-based visual token generation.

Takes Inverse Splat feature maps and produces visual tokens compatible
with Alpamayo-R1-10B (Qwen3-VL-2B-Instruct token space).

Pipeline:
  1. Feature map (36x20x360) -> flatten -> (720, 360)
  2. Linear projection: 360 -> d_model
  3. Learnable query tokens: (180, d_model)
  4. Transformer Decoder: queries cross-attend to feature sequence
  5. Output projection: d_model -> 2048

Reference: /mnt/mydisk/alpamayo/src/alpamayo_r1/models/nuscenes_vision_encoder.py
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class TokenDecoder(nn.Module):
    """Transformer decoder for visual token generation.

    Uses learnable query tokens that cross-attend to flattened feature map
    sequence to produce visual tokens in the target embedding space.

    Args:
        in_channels: Input feature channels from Inverse Splat.
        d_model: Internal transformer dimension.
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        num_query_tokens: Number of output visual tokens per camera.
        output_dim: Output token embedding dimension (must match Qwen3-VL).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 360,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_query_tokens: int = 180,
        output_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_query_tokens = num_query_tokens
        self.output_dim = output_dim

        # Input projection: feature channels -> d_model
        self.input_proj = nn.Linear(in_channels, d_model)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, d_model) * 0.02
        )

        # Positional embeddings
        self.register_buffer(
            "query_pos_embed",
            self._build_sinusoidal_pos(num_query_tokens, d_model),
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        self._init_weights()

        # Storage for student attention maps (populated during forward)
        self.student_attention_maps: list[torch.Tensor] = []

    @staticmethod
    def _build_sinusoidal_pos(num_tokens: int, dim: int) -> torch.Tensor:
        """Build sinusoidal positional embeddings.

        Args:
            num_tokens: Number of tokens.
            dim: Embedding dimension.

        Returns:
            Positional embeddings, shape (1, num_tokens, dim).
        """
        pos = torch.arange(num_tokens).unsqueeze(1).float()
        dim_idx = torch.arange(dim).float()
        div_term = torch.exp(-dim_idx * (math.log(10000.0) / dim))

        pe = torch.zeros(1, num_tokens, dim)
        pe[0, :, 0::2] = torch.sin(pos * div_term[0::2])
        pe[0, :, 1::2] = torch.cos(pos * div_term[1::2])
        return pe

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Output projection with small init
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        feature_map: torch.Tensor,
        capture_attention: bool = False,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate visual tokens from feature map.

        Args:
            feature_map: Feature map, shape (B, C, fH, fW).
            capture_attention: If True, capture and return student attention maps.

        Returns:
            tokens: Visual tokens, shape (B, num_query_tokens, output_dim).
            attention_maps: List of attention maps per layer (empty if not captured).
        """
        B, C, fH, fW = feature_map.shape

        # Flatten spatial dims: (B, C, fH, fW) -> (B, fH*fW, C)
        memory = feature_map.flatten(2).permute(0, 2, 1)  # (B, S, C)

        # Project to d_model
        memory = self.input_proj(memory)  # (B, S, d_model)

        # Add spatial positional embedding to memory
        S = memory.shape[1]
        memory_pos = self._build_sinusoidal_pos(S, self.d_model).to(
            device=memory.device, dtype=memory.dtype
        )
        memory = memory + memory_pos

        # Prepare queries
        queries = self.query_tokens.expand(B, -1, -1)  # (B, Q, d_model)
        queries = queries + self.query_pos_embed.to(device=queries.device, dtype=queries.dtype)

        # Capture attention maps if requested
        attn_maps: list[torch.Tensor] = []
        hooks = []
        if capture_attention:
            for layer in self.decoder.layers:
                hook = layer.multihead_attn.register_forward_hook(
                    self._make_attn_hook(attn_maps)
                )
                hooks.append(hook)

        try:
            # Cross-attention: queries attend to memory
            output = self.decoder(tgt=queries, memory=memory)  # (B, Q, d_model)
        finally:
            for h in hooks:
                h.remove()

        # Output projection
        output = self.output_norm(output)
        tokens = self.output_proj(output)  # (B, Q, output_dim)

        return tokens, attn_maps

    @staticmethod
    def _make_attn_hook(attn_list: list) -> callable:
        """Create forward hook to capture cross-attention weights.

        Args:
            attn_list: List to append captured attention maps to.

        Returns:
            Hook function.
        """
        def hook_fn(
            module: nn.Module,
            input: tuple,
            output: tuple,
        ) -> None:
            """Capture attention weights from MultiheadAttention.

            Args:
                module: The attention module.
                input: Forward input tuple.
                output: Forward output tuple (attn_output, attn_weights).
            """
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_list.append(output[1].detach())
        return hook_fn


class MultiCameraTokenDecoder(nn.Module):
    """Token decoder for multiple target cameras.

    Args:
        in_channels: Input feature channels.
        d_model: Internal transformer dimension.
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        num_query_tokens: Number of output visual tokens per camera.
        output_dim: Output token embedding dimension.
        camera_names: List of target camera names.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 360,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_query_tokens: int = 180,
        output_dim: int = 2048,
        camera_names: list[str] = ("front_wide", "cross_left", "cross_right"),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.camera_names = list(camera_names)

        # Each camera gets its own decoder (tokens are camera-specific)
        self.decoders = nn.ModuleDict({
            cam: TokenDecoder(
                in_channels=in_channels,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                num_query_tokens=num_query_tokens,
                output_dim=output_dim,
                dropout=dropout,
            )
            for cam in self.camera_names
        })

    def forward(
        self,
        feature_maps: Dict[str, torch.Tensor],
        capture_attention: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, list[torch.Tensor]]]:
        """Generate visual tokens for all target cameras.

        Args:
            feature_maps: Dict mapping camera names to feature tensors (B, C, fH, fW).
            capture_attention: Whether to capture student attention maps.

        Returns:
            tokens: Dict mapping camera names to token tensors (B, Q, output_dim).
            attention_maps: Dict mapping camera names to attention map lists.
        """
        all_tokens = {}
        all_attn = {}
        for cam in self.camera_names:
            if cam in feature_maps:
                tokens, attn_maps = self.decoders[cam](
                    feature_maps[cam], capture_attention=capture_attention
                )
                all_tokens[cam] = tokens
                all_attn[cam] = attn_maps
        return all_tokens, all_attn
