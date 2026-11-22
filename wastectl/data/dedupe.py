"""Perceptual (difference) hash dedupe, so near-identical frames can't leak across splits."""
import numpy as np
from PIL import Image


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, hashes):
    """Hamming distance between one 64-bit hash and an array of them."""
    x = np.bitwise_xor(np.uint64(a), np.asarray(hashes, dtype=np.uint64))
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def keep_mask(hashes, max_dist=3):
    """Greedy: walk in the given order (put test first) and drop anything within max_dist of a kept hash."""
    kept, mask = [], []
    for h in hashes:
        dup = bool(kept) and hamming(h, kept).min() <= max_dist
        mask.append(not dup)
        if not dup:
            kept.append(h)
    return mask
