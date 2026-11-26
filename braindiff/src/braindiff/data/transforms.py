"""Volume -> NeuroVFM tokens, done in the dataloader.

Two upstream defaults are overridden: `remove_background` is off (it is per-volume adaptive and gives unequal token
counts across series), and `fixed_scale` (robust foreground window -> [0, 1], after augmentation) replaces
per-volume min-max. Geometry is NeuroVFM's: 1x1x4 mm, patches (4,16,16); the 193x229x193 template -> (48,224,192) ->
12x14x12 = 2016 tokens.
"""
import numpy as np
import torch
from monai.transforms import MapTransform

PATCH = (4, 16, 16)
TOKEN_DIM = PATCH[0] * PATCH[1] * PATCH[2]      # 1024
N_TOKENS = 12 * 14 * 12                          # 2016
INTENSITY_CLIP = (0.0, 1.0)


def fixed_scale(arr, mask=None):
    """Robust-percentile (0.5 / 99.5) clip on the foreground, mapped to [0, 1]."""
    fg = arr[mask] if mask is not None and mask.any() else arr[arr > 0]
    if fg.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(fg, 0.5), np.percentile(fg, 99.5)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), *INTENSITY_CLIP).astype(np.float32)


def to_neurovfm_grid(volume, axcodes):
    """1 mm RAS template volume -> NeuroVFM's [D, H, W] grid: flip, mean-pool groups of 4, crop. RAS -> RPI flips BOTH
    the A and S axes."""
    if tuple(axcodes) != ("R", "A", "S"):
        raise ValueError(f"expected RAS template volumes, got {axcodes}; the fixed downsample hardcodes the RAS->RPI flips")
    v = torch.as_tensor(volume)
    v = torch.flip(v, [1, 2])                    # A->P and S->I
    v = v.permute(2, 1, 0).contiguous()          # (R,A,S) -> (S,A,R) = (D,H,W)
    d, h, w = v.shape
    v = v[: d // PATCH[0] * PATCH[0]]
    v = v.reshape(-1, PATCH[0], h, w).mean(1)    # 1 mm -> 4 mm, area average
    hs, ws = (h % PATCH[1]) // 2, (w % PATCH[2]) // 2      # upstream's start_index = (size % patch) // 2
    return v[:, hs:hs + h // PATCH[1] * PATCH[1], ws:ws + w // PATCH[2] * PATCH[2]]


def tokenize_volume_fast(vol):
    """[D, H, W] -> (tokens [N, 1024], coords [N, 3]) by pure reshape; neurovfm's tokenize_volume layout (row-major over
    (d, h, w), w fastest; each token flattens (c, p_d, p_h, p_w))."""
    d, h, w = vol.shape
    nd, nh, nw = d // PATCH[0], h // PATCH[1], w // PATCH[2]
    tokens = (vol.reshape(nd, PATCH[0], nh, PATCH[1], nw, PATCH[2]).permute(0, 2, 4, 1, 3, 5)
              .reshape(nd * nh * nw, -1))
    coords = torch.stack(torch.meshgrid(torch.arange(nd), torch.arange(nh), torch.arange(nw), indexing="ij"), dim=-1)
    return tokens, coords.reshape(-1, 3)


class NeuroVFMGridd(MapTransform):
    """Path -> [1, 48, 224, 192] on NeuroVFM's grid, channel-first for MONAI (grid -> augment -> tokenize)."""

    def __init__(self, keys, allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        import nibabel as nib
        d = dict(data)
        for key in self.key_iterator(d):
            img = nib.load(str(d[key]))
            d[key] = to_neurovfm_grid(img.get_fdata(dtype=np.float32), nib.aff2axcodes(img.affine)).unsqueeze(0)
        return d


class NeuroVFMTokenize(MapTransform):
    """[1, D, H, W] -> {"tokens": [N, 1024] fp16, "coords": [N, 3] int16}, after augmentation."""

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
            d[key] = {"tokens": tokens.to(torch.float16), "coords": coords.to(torch.int16)}
        return d
