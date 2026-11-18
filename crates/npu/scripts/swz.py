"""swz.py IN OUT [KEY]: swizzle a buffer for NPU37XX CMX (key 1..5, default 5):
the 16-byte blocks of every 512-byte row r rotate by (r mod 2^K) * 2^(5-K)
blocks; the pattern repeats every 512 << K bytes. The length must be a
multiple of that period (the DMA'd tiles start at such offsets)."""
import sys
import numpy as np


def swizzle(b, key=5):
    a = np.frombuffer(b, np.uint8)
    rows = 1 << key
    assert a.size % (512 * rows) == 0, a.size
    v = a.reshape(-1, rows, 32, 16)
    o = np.empty_like(v)
    for r in range(rows):
        o[:, r] = np.roll(v[:, r], r << (5 - key), axis=1)
    return o.tobytes()


if __name__ == "__main__":
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    open(sys.argv[2], "wb").write(swizzle(open(sys.argv[1], "rb").read(), k))
