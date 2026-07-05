import math

import torch
import torch.nn.functional as F


def _extract_patches(x, patch_size, stride):
    """Splits `x` (n_crops, c, h, w) into `patch_size` x `patch_size` patches on a `stride` grid,
    TF-'SAME'-padding first so every pixel is covered by at least one patch regardless of image
    size. Returns (n_crops, c * patch_size**2, num_patches)."""
    _, _, h, w = x.shape
    out_h = math.ceil(h / stride)
    out_w = math.ceil(w / stride)
    pad_h = max((out_h - 1) * stride + patch_size - h, 0)
    pad_w = max((out_w - 1) * stride + patch_size - w, 0)
    x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))  # (n_crops, c, h + pad_h, w + pad_w)
    return F.unfold(x, patch_size, stride=stride)  # (n_crops, c * patch_size**2, out_h * out_w)


def _resize_preserve_aspect_ratio(image, longer_side_length):
    _, _, h, w = image.shape
    ratio = longer_side_length / max(h, w)
    resized_h = round(h * ratio)
    resized_w = round(w * ratio)
    resized = F.interpolate(image, (resized_h, resized_w), mode="bicubic", align_corners=False)  # (n_crops, c, resized_h, resized_w)
    return resized, resized_h, resized_w


def _hashed_spatial_position_indices(grid_size, count_h, count_w):
    """Hashes a count_h x count_w patch grid down onto a grid_size x grid_size grid — see
    embeddings.HashedSpatialPositionEmbedding for why. Returns a (1, count_h * count_w) index
    tensor, one grid cell in [0, grid_size**2) per patch, in row-major patch order."""
    grid = torch.arange(grid_size, dtype=torch.float32)  # (grid_size,)

    col_hash = F.interpolate(grid.reshape(1, 1, grid_size), size=count_w, mode="nearest")  # (1, 1, count_w)
    col_hash = col_hash.repeat(1, count_h, 1)  # (1, count_h, count_w)

    row_hash = F.interpolate(grid.reshape(1, 1, grid_size), size=count_h, mode="nearest")  # (1, 1, count_h)
    row_hash = row_hash.transpose(1, 2).repeat(1, 1, count_w)  # (1, count_h, count_w)

    return (row_hash * grid_size + col_hash).reshape(1, -1)  # (1, count_h * count_w)


def _pad_or_truncate(x, max_seq_len):
    """x: (n_crops, c, num_patches). max_seq_len < 0 means keep every patch, unmodified."""
    if max_seq_len < 0:
        return x
    n_crops, c, num_patches = x.shape
    if num_patches >= max_seq_len:
        return x[:, :, :max_seq_len]  # (n_crops, c, max_seq_len)
    padding = x.new_zeros(n_crops, c, max_seq_len - num_patches)
    return torch.cat([x, padding], dim=-1)  # (n_crops, c, max_seq_len)


def _patches_with_positions(image, patch_size, patch_stride, hse_grid_size, scale_id, max_seq_len):
    """Returns (n_crops, c * patch_size**2 + 3, seq_len): patches, each one's hashed spatial-position
    index, its scale index, and a validity mask (1 for real patches, 0 for padding), stacked along
    the feature dim so they can all ride through padding/truncation together."""
    n_crops, _, h, w = image.shape
    patches = _extract_patches(image, patch_size, patch_stride)  # (n_crops, c * patch_size**2, num_patches)

    count_h = math.ceil(h / patch_stride)
    count_w = math.ceil(w / patch_stride)
    spatial_idx = _hashed_spatial_position_indices(hse_grid_size, count_h, count_w).to(patches)  # (1, num_patches)
    spatial_idx = spatial_idx.unsqueeze(1).repeat(n_crops, 1, 1)  # (n_crops, 1, num_patches)
    scale_idx = torch.full_like(spatial_idx, scale_id)
    mask = torch.ones_like(spatial_idx)

    out = torch.cat([patches, spatial_idx, scale_idx, mask], dim=1)  # (n_crops, c * patch_size**2 + 3, num_patches)
    return _pad_or_truncate(out, max_seq_len)  # (n_crops, c * patch_size**2 + 3, max_seq_len)


def get_multiscale_patches(image, patch_size, patch_stride, hse_grid_size, longer_side_lengths,
                            max_seq_len_from_original_res):
    """Builds MUSIQ's multi-scale patch sequence from a batch of images: for each length in
    `longer_side_lengths`, resizes (preserving aspect ratio, longer side -> that length) and splits
    into patches; optionally appends patches from the native resolution too
    (`max_seq_len_from_original_res` >= 0, or < 0 for "every native-resolution patch, unpadded"; None
    skips the native-resolution scale entirely). Every scale's patches are packed with a hashed
    spatial-position index, a scale index, and a validity mask (see _patches_with_positions), then
    concatenated end-to-end into one token sequence.

    Args:
        image: (n_crops, 3, h, w) — a batch of same-sized images (or crops of one image).
    Returns:
        (n_crops, total_seq_len, 3 * patch_size**2 + 3)

    Faithful reimplementation of pyiqa's get_multiscale_patches
    (pyiqa/data/multiscale_trans_util.py), which itself ports MUSIQ's original TF preprocessing.py.
    """
    longer_side_lengths = sorted(longer_side_lengths)

    scales = []
    for scale_id, longer_side_length in enumerate(longer_side_lengths):
        resized, _, _ = _resize_preserve_aspect_ratio(image, longer_side_length)
        max_seq_len = math.ceil(longer_side_length / patch_stride) ** 2
        scales.append(_patches_with_positions(
            resized, patch_size, patch_stride, hse_grid_size, scale_id, max_seq_len,
        ))

    if max_seq_len_from_original_res is not None:
        scales.append(_patches_with_positions(
            image, patch_size, patch_stride, hse_grid_size, len(longer_side_lengths),
            max_seq_len_from_original_res,
        ))

    stacked = torch.cat(scales, dim=-1)  # (n_crops, 3 * patch_size**2 + 3, total_seq_len)
    return stacked.transpose(1, 2)  # (n_crops, total_seq_len, 3 * patch_size**2 + 3)
