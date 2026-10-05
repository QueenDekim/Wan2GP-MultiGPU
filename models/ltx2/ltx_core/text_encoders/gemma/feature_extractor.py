import math

import torch
from einops import rearrange

from ...model.model_protocol import ModelConfigurator


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    b, t, d, l = encoded_text.shape
    token_indices = torch.arange(t, device=encoded_text.device)[None, :]
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")
    eps = 1e-6
    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)
    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min
    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, t, -1)
    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    return normed.masked_fill(~mask_flattened, 0.0)


def _norm_and_concat_per_token_rms(encoded_text: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    b, t, d, l = encoded_text.shape
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(b, t, d * l)
    mask_3d = attention_mask.bool().unsqueeze(-1)
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    return x * math.sqrt(target_dim / source_dim)


class GemmaFeaturesExtractorProjLinear(torch.nn.Module, ModelConfigurator["GemmaFeaturesExtractorProjLinear"]):
    """Supports both 19B (single projection) and 22B/2.3 (dual projections)."""

    def __init__(
        self,
        aggregate_embed: torch.nn.Linear | None = None,
        video_aggregate_embed: torch.nn.Linear | None = None,
        audio_aggregate_embed: torch.nn.Linear | None = None,
        embedding_dim: int = 3840,
    ) -> None:
        super().__init__()
        self.aggregate_embed = aggregate_embed
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    @property
    def is_v2(self) -> bool:
        return self.video_aggregate_embed is not None

    def _projection_device(self) -> torch.device:
        """Return the CUDA device that owns the active projection weights.

        With Accelerate MultiGPU, Gemma hidden states are produced on the GPU
        that owns each decoder layer. The feature extractor, however, flattens
        all 49 states into one vector and therefore requires them on one device.
        Gather activations to the projection weight's device, never to a
        hard-coded cuda:0.
        """
        for projection in (
            self.video_aggregate_embed,
            self.aggregate_embed,
            self.audio_aggregate_embed,
        ):
            if projection is None:
                continue
            weight = getattr(projection, "weight", None)
            device = getattr(weight, "device", None)
            if device is not None and torch.device(device).type != "meta":
                return torch.device(device)
        return torch.device("cuda", torch.cuda.current_device())

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...] | torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        target_device = self._projection_device()
        attention_mask = attention_mask.to(target_device, non_blocking=True)

        if isinstance(hidden_states, (list, tuple)):
            # Decoder-layer outputs follow the Gemma shard placement, so a
            # tuple can legitimately contain tensors from cuda:0, cuda:1, ...
            # Only activations are gathered here; model weights remain sharded.
            gathered = [
                state if state.device == target_device
                else state.to(target_device, non_blocking=True)
                for state in hidden_states
            ]
            encoded = torch.stack(gathered, dim=-1)
            del gathered
        else:
            encoded = (
                hidden_states
                if hidden_states.device == target_device
                else hidden_states.to(target_device, non_blocking=True)
            )

        if self.is_v2:
            normed = _norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)
            v_dim = self.video_aggregate_embed.out_features
            video = self.video_aggregate_embed(_rescale_norm(normed, v_dim, self.embedding_dim))
            audio = None
            if self.audio_aggregate_embed is not None:
                a_dim = self.audio_aggregate_embed.out_features
                audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))
            return video, audio

        if self.aggregate_embed is None:
            raise ValueError("Feature extractor is missing aggregate projection weights.")
        sequence_lengths = attention_mask.sum(dim=-1)
        normed = _norm_and_concat_padded_batch(encoded, sequence_lengths, padding_side=padding_side)
        features = self.aggregate_embed(normed.to(encoded.dtype))
        return features, features

    @classmethod
    def from_config(cls: type["GemmaFeaturesExtractorProjLinear"], config: dict) -> "GemmaFeaturesExtractorProjLinear":
        transformer_config = config.get("transformer", {})
        embedding_dim = 3840
        num_layers_plus_embed = 49
        flat_dim = embedding_dim * num_layers_plus_embed
        if transformer_config.get("caption_proj_before_connector", False):
            video_inner_dim = transformer_config.get("num_attention_heads", 32) * transformer_config.get(
                "attention_head_dim", 128
            )
            audio_inner_dim = transformer_config.get("audio_num_attention_heads", 32) * transformer_config.get(
                "audio_attention_head_dim", 64
            )
            return cls(
                aggregate_embed=None,
                video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
                audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
                embedding_dim=embedding_dim,
            )

        return cls(
            aggregate_embed=torch.nn.Linear(flat_dim, embedding_dim, bias=False),
            video_aggregate_embed=None,
            audio_aggregate_embed=None,
            embedding_dim=embedding_dim,
        )
