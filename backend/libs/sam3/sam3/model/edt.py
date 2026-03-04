# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Euclidean distance transform (EDT) - with triton or pure PyTorch fallback"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _edt_pytorch(data: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch fallback for EDT when triton is unavailable (e.g. Windows).

    Uses iterative approximation via max-pooling distance propagation.
    Not as precise as the O(N^2) triton kernel but sufficient for SAM3's
    interactive point sampling use case.
    """
    assert data.dim() == 3
    B, H, W = data.shape
    dist = torch.where(data.bool(), torch.tensor(float("inf"), device=data.device), torch.tensor(0.0, device=data.device))

    max_dist = H + W
    for _ in range(max_dist):
        padded = torch.nn.functional.pad(dist, (1, 1, 1, 1), value=float("inf"))
        neighbors = torch.stack([
            padded[:, 1:-1, 1:-1],     # center
            padded[:, 0:-2, 1:-1] + 1, # up
            padded[:, 2:,   1:-1] + 1, # down
            padded[:, 1:-1, 0:-2] + 1, # left
            padded[:, 1:-1, 2:]   + 1, # right
        ], dim=0)
        new_dist = neighbors.min(dim=0).values
        if torch.equal(new_dist, dist):
            break
        dist = new_dist

    return dist.sqrt()


def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Uses triton kernel on Linux, falls back to pure PyTorch on Windows.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
    """
    if not HAS_TRITON:
        return _edt_pytorch(data)

    assert data.dim() == 3
    assert data.is_cuda
    B, H, W = data.shape
    data = data.contiguous()

    output = torch.where(data, 1e18, 0.0)
    assert output.is_contiguous()

    parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
    parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, H)

    edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=True,
    )

    parabola_loc.zero_()
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, W)
    edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=False,
    )
    return output.sqrt()


if HAS_TRITON:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)
