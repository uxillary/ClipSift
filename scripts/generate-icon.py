"""Generate ClipSift's simple multi-resolution Windows icon."""

from pathlib import Path

from PIL import Image, ImageDraw


SIZES = (16, 32, 48, 256)
OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "clipsift.ico"


def make_icon(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (23, 26, 29, 255))
    draw = ImageDraw.Draw(image)
    width = max(2, round(size * 0.117))
    padding = round(size * 0.2)
    draw.arc((padding, padding, size - padding, size - padding), 45, 315, fill=(69, 196, 168, 255), width=width)
    middle = size // 2
    draw.line(
        (round(size * 0.30), middle, round(size * 0.68), middle),
        fill=(215, 224, 217, 255),
        width=max(2, round(size * 0.094)),
    )
    radius = max(1, round(size * 0.074))
    centre = round(size * 0.715)
    draw.ellipse((centre - radius, middle - radius, centre + radius, middle + radius), fill=(69, 196, 168, 255))
    return image


if __name__ == "__main__":
    frames = [make_icon(size) for size in SIZES]
    frames[-1].save(OUTPUT, format="ICO", sizes=[(size, size) for size in SIZES])
