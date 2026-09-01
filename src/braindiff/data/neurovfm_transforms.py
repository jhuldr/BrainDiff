"""Volume -> NeuroVFM tokens, done in the dataloader.

NeuroVFM is a packed variable-length token model: it consumes `img [N, 1024]` +
`coords [N, 3]`, not a dense volume. Tokenizing is pure CPU work (SimpleITK resample
dominates at ~1.1 s/volume), so it belongs in the worker processes.

Two upstream defaults are overridden:

1. `remove_background=True` drops a token if any voxel in its 4x16x16 patch falls below a
   per-volume 10th-percentile threshold, giving unequal token counts and index sets across
   series (measured 1034/1029/1034/1034 for one subject). That makes a token-wise ref/main
   difference meaningless, so it is off and every volume yields the full dense grid.
2. `prepare_for_inference` rescales each volume by its own percentile clip and min-max, so
   two timepoints get different affine maps and the difference mixes anatomy with intensity
   rescaling. One fixed affine is used for every volume instead.

Geometry is NeuroVFM's: 1x1x4 mm, patches (4,16,16). The 193x229x193 @ 1 mm template lands
on [D,H,W] = (48,224,192) -> a 12x14x12 = 2016 token grid at ~16 mm per token.
"""
import numpy as np
import torch
from monai.transforms import MapTransform

PATCH = (4, 16, 16)
TOKEN_DIM = PATCH[0] * PATCH[1] * PATCH[2]      # 1024
# 12x14x12 grid from the S4 template FOV at 1x1x4 mm.
N_TOKENS = 12 * 14 * 12                          # 2016

# Fixed intensity window, shared by every volume. Percentiles of the nonzero
# voxels across the S4 template volumes; a fixed map is what keeps the ref/main
# difference about anatomy rather than about per-volume rescaling.
INTENSITY_CLIP = (0.0, 1.0)


def fixed_scale(arr, mask=None):
    """One affine for all volumes: robust-percentile clip on the foreground, then
    map that window to [0, 1]. Unlike upstream's per-volume min-max, the same
    intensity means the same thing in both timepoints."""
    fg = arr[mask] if mask is not None and mask.any() else arr[arr > 0]
    if fg.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(fg, 0.5), np.percentile(fg, 99.5)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), *INTENSITY_CLIP).astype(np.float32)


def to_neurovfm_grid(volume, axcodes):
    """1 mm isotropic RAS template volume -> NeuroVFM's [D, H, W] grid.

Every volume here is already template-space with an identical affine, so the resample is
one fixed downsample rather than a general one: flip, mean-pool groups of 4, crop. Upstream
does it with SimpleITK at 460 ms/volume, 89% of the preprocessing budget.

RAS -> RPI flips BOTH the A and S axes; omitting them scores 0.58 voxel correlation against
the reference, including them 0.9935 (residual is BSpline vs mean-pool interpolation).
    """
    if tuple(axcodes) != ("R", "A", "S"):
        raise ValueError(
            f"expected RAS template volumes, got {axcodes}. The fixed downsample "
            f"below hardcodes the RAS->RPI flips; a different orientation needs "
            f"its own flip set, so failing loudly rather than silently misaligning."
        )
    v = torch.as_tensor(volume)
    v = torch.flip(v, [1, 2])                    # A->P and S->I
    v = v.permute(2, 1, 0).contiguous()          # (R,A,S) -> (S,A,R) = (D,H,W)

    d, h, w = v.shape
    v = v[: d // PATCH[0] * PATCH[0]]
    v = v.reshape(-1, PATCH[0], h, w).mean(1)    # 1 mm -> 4 mm, area average
    # Centre-crop in-plane to patch multiples, matching upstream's
    # start_index = (size % patch) // 2.
    hs, ws = (h % PATCH[1]) // 2, (w % PATCH[2]) // 2
    return v[:, hs:hs + h // PATCH[1] * PATCH[1], ws:ws + w // PATCH[2] * PATCH[2]]


def neurovfm_geometry(orig_shape):
    """Per-axis (flip, offset, extent) taking a native RAS voxel index to the NeuroVFM grid, in
output axis order (D, H, W).

The single source of truth for how a volume maps into the preprocessed image, shared by
`to_neurovfm_grid` (which builds the image) and `voxel_box_to_percent` (which places box
coordinates in it); both must derive from the same formulas.

Returns (native_axis, flip, offset, extent) in D, H, W order:
  D <- native axis 2 (S), flipped, mean-pooled by 4
  H <- native axis 1 (A), flipped, centre-cropped to a multiple of 16
  W <- native axis 0 (R), not flipped, centre-cropped to a multiple of 16
    """
    R, A, S = orig_shape
    return [
        (2, True,  0,                 S // PATCH[0]),           # D, after //4 pooling
        (1, True,  (A % PATCH[1]) // 2, A // PATCH[1] * PATCH[1]),  # H
        (0, False, (R % PATCH[2]) // 2, R // PATCH[2] * PATCH[2]),  # W
    ]


def voxel_box_to_percent(box, orig_shape=(193, 229, 193)):
    """Native-voxel box -> six ints 0..100, in the NeuroVFM frame.

Input is (x1, x2, y1, y2, z1, z2), raw np.argwhere min/max indices in native axis-major
order (x=R, y=A, z=S). Output is each coordinate's position in the final preprocessed image
as a percentage of that image's extent, so the model needs no resample affine.

Note D=S and W=R, so the first and third axes swap relative to a naive x->D, y->H, z->W
mapping, and the RAS->RPI reorientation mirrors two of them. A box normalized under the
wrong convention is off by up to 66 percentage points once off-centre.
    """
    lohi = [(box[0], box[1]), (box[2], box[3]), (box[4], box[5])]   # per native axis
    out = []
    for native_axis, flip, offset, extent in neurovfm_geometry(orig_shape):
        lo, hi = lohi[native_axis]
        n = orig_shape[native_axis]
        if flip:
            lo, hi = n - 1 - hi, n - 1 - lo
        if native_axis == 2:                      # D: 4 native slices -> 1 output slice
            lo, hi = lo / PATCH[0], hi / PATCH[0]
        else:
            lo, hi = lo - offset, hi - offset
        for c in (lo, hi):
            out.append(int(round(c / max(1, extent - 1) * 100)))

    # Clamp, and keep each axis non-degenerate so a box never collapses to a
    # zero-width interval after rounding.
    for i in range(0, 6, 2):
        out[i] = max(0, min(100, out[i]))
        out[i + 1] = max(0, min(100, out[i + 1]))
        if out[i + 1] <= out[i]:
            out[i + 1] = min(100, out[i] + 1)
            if out[i + 1] <= out[i]:
                out[i] = max(0, out[i + 1] - 1)
    return tuple(out)


def tokenize_volume_fast(vol):
    """[D, H, W] -> (tokens [N, 1024], coords [N, 3]) by pure reshape.

    Same layout as neurovfm's tokenize_volume: tokens are row-major over
    (d, h, w) with w fastest, and each token flattens (c, p_d, p_h, p_w).
    """
    d, h, w = vol.shape
    nd, nh, nw = d // PATCH[0], h // PATCH[1], w // PATCH[2]
    tokens = (vol.reshape(nd, PATCH[0], nh, PATCH[1], nw, PATCH[2])
                 .permute(0, 2, 4, 1, 3, 5)
                 .reshape(nd * nh * nw, -1))
    coords = torch.stack(torch.meshgrid(torch.arange(nd), torch.arange(nh),
                                        torch.arange(nw), indexing="ij"), dim=-1)
    return tokens, coords.reshape(-1, 3)


def tokenize_path(path):
    """NIfTI path -> (tokens [N, 1024] float16, coords [N, 3] int16)."""
    import nibabel as nib

    img = nib.load(str(path))
    vol = to_neurovfm_grid(img.get_fdata(dtype=np.float32), nib.aff2axcodes(img.affine))
    vol = torch.from_numpy(fixed_scale(vol.numpy()))
    if min(vol.shape) < max(PATCH):
        return None, None
    tokens, coords = tokenize_volume_fast(vol)
    return tokens.to(torch.float16), coords.to(torch.int16)


class NeuroVFMGridd(MapTransform):
    """Path -> [1, 48, 224, 192] on NeuroVFM's grid, channel-first for MONAI.

Kept separate from tokenization so augmentation still has a spatial volume to work on:
grid -> augment -> tokenize. Augmenting a flat token list is not the same operation.
    """

    def __init__(self, keys, allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        import nibabel as nib
        d = dict(data)
        for key in self.key_iterator(d):
            img = nib.load(str(d[key]))
            vol = to_neurovfm_grid(img.get_fdata(dtype=np.float32),
                                   nib.aff2axcodes(img.affine))
            d[key] = vol.unsqueeze(0)                     # [1, D, H, W]
        return d


class NeuroVFMTokenize(MapTransform):
    """[1, D, H, W] -> {"tokens": [N, 1024], "coords": [N, 3]}, after augmentation."""

    def __init__(self, keys, allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            vol = torch.as_tensor(d[key])
            if vol.ndim == 4:
                vol = vol[0]
            vol = torch.from_numpy(fixed_scale(np.asarray(vol, dtype=np.float32)))
            tokens, coords = tokenize_volume_fast(vol)
            d[key] = {"tokens": tokens.to(torch.float16),
                      "coords": coords.to(torch.int16)}
        return d


def stack_modalities(per_modality, n_tokens):
    """[{tokens, coords} or None] * M -> (tokens [M,N,1024], coords [M,N,3], present [M]).

Shapes are uniform because every volume is template-space, which lets the encoder pack a
batch and reshape the output back to [B, N, D].
    """
    m = len(per_modality)
    tokens = torch.zeros(m, n_tokens, TOKEN_DIM, dtype=torch.float16)
    coords = torch.zeros(m, n_tokens, 3, dtype=torch.int16)
    present = torch.zeros(m, dtype=torch.bool)
    for i, entry in enumerate(per_modality):
        if entry is None:
            continue
        if entry["tokens"].shape[0] != n_tokens:
            continue                                      # off-grid volume, drop it
        tokens[i], coords[i], present[i] = entry["tokens"], entry["coords"], True
    return tokens, coords, present
