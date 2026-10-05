from typing import Any

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6, in_place = False) -> torch.Tensor:
    # deepbeepmeep RMS Norm
    if in_place:
        scale = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True, dtype=torch.float32)
        scale.square_().div_(x.shape[-1]).add_(eps).rsqrt_()
        x.mul_(scale)
        if weight is not None:
            x.mul_(weight)
        return x
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def check_config_value(config: dict, key: str, expected: Any) -> None:  # noqa: ANN401
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity.
    Returns:
        Velocity
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Returns:
        Denoised sample
    """
    # MultiGPU transformer sharding may produce velocity on the final shard
    # while the latent/sample remains on the pipeline primary device. Return
    # denoised data on the sample device so the rest of the pipeline keeps a
    # stable owner and only one cross-GPU copy is needed after the last block.
    target_device = sample.device
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(device=target_device, dtype=calc_dtype, non_blocking=True)
    sample_calc = sample.to(dtype=calc_dtype)
    velocity_calc = velocity.to(device=target_device, dtype=calc_dtype, non_blocking=True)
    return (sample_calc - velocity_calc * sigma).to(sample.dtype)
