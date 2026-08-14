"""Preserve ambient Torch sampling semantics across compiled LatentSeq execution.

LatentSeq never owns random-number-generator state.  These operators exist solely because
`torch.compile` may lower some random Torch functions to different random kernels than eager
execution.  Marking the configured sampling call as an opaque Torch operator keeps the same ambient
Torch generator and the same underlying eager operation in both modes without storing, seeding,
snapshotting, or restoring RNG state.
"""

import json
from collections.abc import Callable

import torch
from torch import Tensor


# helpers


def call_torch_sampler(
    function: Callable,
    params: dict[str, object],
    batch_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Call a configured Torch sampler with a batch-shaped request.

    Args:
        function: Torch callable selected by configuration.
        params: Keyword parameters supplied by that configuration.
        batch_width: Requested one-dimensional sample width.
        device: Device on which the sample must be produced.

    Returns:
        Tensor returned by the configured Torch function.  Shape and value legality are validated
        by the construction boundary that owns the configuration.
    """
    try:
        result = function(size=(batch_width,), device=device, **params)
    except TypeError:
        try:
            result = function((batch_width,), device=device, **params)
        except TypeError as error:
            raise ValueError(
                "configured dwell sampler does not accept a batch-shaped request"
            ) from error
    if not isinstance(result, torch.Tensor):
        raise ValueError("configured Torch sampler must return a Tensor")
    return result


# main


@torch.library.custom_op("latentseq::ambient_randint", mutates_args=())
def _ambient_randint(
    device_anchor: Tensor,
    low: int,
    high: int,
    batch_width: int,
) -> Tensor:
    """Execute `torch.randint` opaquely while consuming only the ambient Torch RNG stream."""
    return torch.randint(
        low,
        high,
        (batch_width,),
        dtype=torch.int64,
        device=device_anchor.device,
    )


@_ambient_randint.register_fake
def _ambient_randint_fake(
    device_anchor: Tensor,
    low: int,
    high: int,
    batch_width: int,
) -> Tensor:
    return torch.empty((batch_width,), dtype=torch.int64, device=device_anchor.device)


@torch.library.custom_op("latentseq::ambient_multinomial", mutates_args=())
def _ambient_multinomial(probabilities: Tensor) -> Tensor:
    """Draw one categorical sample per row using the ambient Torch RNG stream."""
    return torch.multinomial(probabilities, num_samples=1).squeeze(1).to(torch.int64)


@_ambient_multinomial.register_fake
def _ambient_multinomial_fake(probabilities: Tensor) -> Tensor:
    return torch.empty(
        (probabilities.shape[0],), dtype=torch.int64, device=probabilities.device
    )


@torch.library.custom_op("latentseq::ambient_configured_sampler", mutates_args=())
def _ambient_configured_sampler(
    device_anchor: Tensor,
    batch_width: int,
    function_name: str,
    params_json: str,
) -> Tensor:
    """Execute one configured Torch sampler opaquely against ambient RNG state."""
    function = getattr(torch, function_name)
    params = json.loads(params_json)
    return call_torch_sampler(
        function,
        params,
        batch_width,
        device_anchor.device,
    ).to(dtype=torch.int64)


@_ambient_configured_sampler.register_fake
def _ambient_configured_sampler_fake(
    device_anchor: Tensor,
    batch_width: int,
    function_name: str,
    params_json: str,
) -> Tensor:
    return torch.empty((batch_width,), dtype=torch.int64, device=device_anchor.device)


# construction


def build_configured_sampler(
    function_name: str,
    params: dict[str, object],
    device_anchor: torch.Tensor,
) -> Callable[[int], torch.Tensor]:
    """Build a compile-safe callback for an already validated Torch sampling configuration.

    Args:
        function_name: Name resolved against `torch` during configuration validation.
        params: JSON-serializable parameters passed to the configured Torch function.
        device_anchor: Tensor whose device identifies where samples must be produced.

    Returns:
        Callback accepting `batch_width`.  It consumes the ordinary ambient Torch RNG stream and
        retains the configured Torch function's eager semantics under `torch.compile`.
    """
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))

    def sample(batch_width: int) -> torch.Tensor:
        return _ambient_configured_sampler(
            device_anchor,
            batch_width,
            function_name,
            params_json,
        )

    return sample


def sample_uniform_integers(
    device_anchor: torch.Tensor,
    low: int,
    high: int,
    batch_width: int,
) -> torch.Tensor:
    """Sample uniform integers without introducing package-owned RNG state."""
    return _ambient_randint(device_anchor, low, high, batch_width)


def sample_categorical(probabilities: torch.Tensor) -> torch.Tensor:
    """Sample one category per probability row without package-owned RNG state."""
    return _ambient_multinomial(probabilities)
