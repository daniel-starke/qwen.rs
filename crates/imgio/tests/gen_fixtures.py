"""Regenerate crates/imgio/tests/data/.

Run from the repo root with a Python that has Pillow:

    python crates/imgio/tests/gen_fixtures.py

Every fixture is paired with a '.ref.ppm' holding **Pillow's own decode** of the
same file, so the tests compare our decoder against a reference implementation
rather than against itself. Regenerating is only needed when adding a case; the
checked-in data is what the tests assert against.
"""
import os, struct
from PIL import Image

OUT = os.path.join("crates", "imgio", "tests", "data")
os.makedirs(OUT, exist_ok=True)

W, H = 37, 23

def pattern(w, h):
    px = bytearray()
    for y in range(h):
        for x in range(w):
            r = (x * 7 + y * 3) % 256
            g = (x * x + y * 11) % 256
            b = (255 - (x * 5 + y * 13) % 256)
            px += bytes((r, g, b))
    return bytes(px)

rgb = Image.frombytes("RGB", (W, H), pattern(W, H))

def save_ref(name, im):
    im = im.convert("RGB")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(("P6\n%d %d\n255\n" % im.size).encode())
        f.write(im.tobytes())

# --- PNG variants -----------------------------------------------------------
rgb.save(os.path.join(OUT, "rgb8.png"))
save_ref("rgb8.ref.ppm", rgb)

gray = rgb.convert("L")
gray.save(os.path.join(OUT, "gray8.png"))
save_ref("gray8.ref.ppm", gray)

pal = rgb.convert("P", palette=Image.ADAPTIVE, colors=64)
pal.save(os.path.join(OUT, "pal8.png"))
save_ref("pal8.ref.ppm", pal)

rgba = rgb.convert("RGBA")
rgba.save(os.path.join(OUT, "rgba8.png"))
save_ref("rgba8.ref.ppm", rgba)   # alpha is opaque, so dropping it == convert

# 1-bit and 4-bit palette PNGs exercise the sub-byte sample path.
bw = gray.convert("1")
bw.save(os.path.join(OUT, "bw1.png"), bits=1)
save_ref("bw1.ref.ppm", bw)

# --- BMP / PPM --------------------------------------------------------------
rgb.save(os.path.join(OUT, "rgb8.bmp"))
save_ref("rgb8_bmp.ref.ppm", rgb)
rgb.save(os.path.join(OUT, "rgb8.ppm"))

# --- JPEG -------------------------------------------------------------------
rgb.save(os.path.join(OUT, "rgb8_420.jpg"), quality=92, subsampling=2)
save_ref("rgb8_420.ref.ppm", Image.open(os.path.join(OUT, "rgb8_420.jpg")))

rgb.save(os.path.join(OUT, "rgb8_444.jpg"), quality=95, subsampling=0)
save_ref("rgb8_444.ref.ppm", Image.open(os.path.join(OUT, "rgb8_444.jpg")))

gray.save(os.path.join(OUT, "gray8.jpg"), quality=95)
save_ref("gray8.ref.jpg.ppm", Image.open(os.path.join(OUT, "gray8.jpg")))

# restart-interval variant
rgb.save(os.path.join(OUT, "rgb8_rst.jpg"), quality=90, subsampling=2, restart_marker_blocks=1)
save_ref("rgb8_rst.ref.ppm", Image.open(os.path.join(OUT, "rgb8_rst.jpg")))

# Progressive JPEG. The hand-written decoder refused SOF2; the `image` crate
# decodes it, so this fixture pins that it is decoded *correctly* rather than
# merely accepted.
rgb.save(os.path.join(OUT, "rgb8_prog.jpg"), quality=92, subsampling=2, progressive=True)
save_ref("rgb8_prog.ref.ppm", Image.open(os.path.join(OUT, "rgb8_prog.jpg")))

# --- resize reference -------------------------------------------------------
big = Image.frombytes("RGB", (160, 120), pattern(160, 120))
big.save(os.path.join(OUT, "resize_src.ppm"))
save_ref("resize_down.ref.ppm", big.resize((32, 24), Image.BICUBIC))
save_ref("resize_up.ref.ppm", big.resize((200, 150), Image.BICUBIC))

for n in sorted(os.listdir(OUT)):
    print(n, os.path.getsize(os.path.join(OUT, n)))


# --- smooth-source resize references ----------------------------------------
# Smooth, overshoot-free source: no hard discontinuities, so Pillow's clipped
# uint8 intermediate cannot diverge from our f32 one.
w,h=160,120
px=bytearray()
for y in range(h):
    for x in range(w):
        px += bytes(((x*255)//(w-1), (y*255)//(h-1), ((x+y)*255)//(w+h-2)))
smooth = Image.frombytes("RGB",(w,h),bytes(px))
smooth.save(os.path.join(OUT,"smooth_src.ppm"))
save_ref("smooth_up.ref.ppm", smooth.resize((200,150), Image.BICUBIC))
save_ref("smooth_down.ref.ppm", smooth.resize((37,29), Image.BICUBIC))
print("ok")


# ---------------------------------------------------------------------------
# Adam7-interlaced PNG.
#
# Pillow cannot *write* interlaced PNG, so this one is encoded by hand: same
# pixels as `rgb8.png`, IHDR interlace method 1, the seven Adam7 passes each
# filtered with filter type 0. Its reference is `rgb8.ref.ppm`: an interlaced
# PNG must decode to exactly the same image as the non-interlaced one, which is
# the whole point of the test.
def write_adam7(path, img):
    import struct, zlib

    w, h = img.size
    px = img.convert("RGB").load()
    # (x0, y0, dx, dy) per Adam7 pass.
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
              (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    raw = bytearray()
    for x0, y0, dx, dy in passes:
        cols = len(range(x0, w, dx))
        if cols == 0:
            continue
        for y in range(y0, h, dy):
            raw.append(0)  # filter: None
            for x in range(x0, w, dx):
                raw.extend(px[x, y])

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 1)  # 8-bit RGB, interlace 1
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        f.write(chunk(b"IEND", b""))


write_adam7(os.path.join(OUT, "rgb8_adam7.png"), rgb)
