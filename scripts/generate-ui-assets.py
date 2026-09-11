"""Generate transparent GUI assets from ClipSift's approved logo."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MASTER = ASSETS / "icon.png"
TEAL = (69, 196, 168, 255)
AMBER = (215, 168, 75, 255)


def logo_derivative(size: int, output: str, opacity: int = 255) -> None:
    with Image.open(MASTER) as source:
        logo = source.convert("RGBA")
    alpha_box = logo.getchannel("A").getbbox()
    if alpha_box:
        logo = logo.crop(alpha_box)
    logo.thumbnail((size, size), Image.Resampling.LANCZOS)
    if opacity != 255:
        logo.putalpha(logo.getchannel("A").point(lambda value: value * opacity // 255))
    canvas = Image.new("RGBA", (size, size))
    canvas.alpha_composite(logo, ((size - logo.width) // 2, (size - logo.height) // 2))
    canvas.save(ASSETS / output)


def ui_icon(name: str, painter: Callable[[ImageDraw.ImageDraw], None]) -> None:
    scale = 3
    canvas = Image.new("RGBA", (18 * scale, 18 * scale))
    painter(ImageDraw.Draw(canvas))
    canvas.resize((18, 18), Image.Resampling.LANCZOS).save(ASSETS / f"ui-{name}-18.png")


def line(draw: ImageDraw.ImageDraw, points, fill=TEAL, width=5) -> None:
    draw.line(points, fill=fill, width=width, joint="curve")


if __name__ == "__main__":
    logo_derivative(36, "brand-header-36.png")
    logo_derivative(96, "brand-watermark-96.png", opacity=78)

    ui_icon("browse", lambda d: (d.rounded_rectangle((5, 16, 49, 45), 5, outline=TEAL, width=5), line(d, ((8, 17), (22, 17), (27, 11), (45, 11)))))
    ui_icon("start", lambda d: d.polygon(((16, 9), (46, 27), (16, 45)), fill=TEAL))
    ui_icon("cancel", lambda d: (line(d, ((13, 13), (41, 41)), AMBER, 6), line(d, ((41, 13), (13, 41)), AMBER, 6)))
    ui_icon("results", lambda d: (d.rounded_rectangle((5, 16, 49, 45), 5, outline=TEAL, width=5), line(d, ((27, 35), (27, 7)), width=5), line(d, ((18, 16), (27, 7), (36, 16)), width=5)))
    ui_icon("check", lambda d: (d.ellipse((7, 7, 47, 47), outline=TEAL, width=5), line(d, ((17, 28), (24, 35), (38, 19)), width=6)))
    ui_icon("about", lambda d: (d.ellipse((7, 7, 47, 47), outline=TEAL, width=5), d.ellipse((25, 15, 30, 20), fill=TEAL), line(d, ((27, 25), (27, 39)), width=5)))
    ui_icon("evidence", lambda d: (d.rounded_rectangle((6, 8, 48, 46), 4, outline=TEAL, width=5), d.ellipse((13, 14, 21, 22), fill=TEAL), d.polygon(((10, 40), (22, 28), (29, 35), (36, 25), (45, 40)), fill=TEAL)))
    ui_icon("video", lambda d: (d.rounded_rectangle((5, 13, 36, 42), 5, outline=TEAL, width=5), d.polygon(((37, 21), (49, 15), (49, 40), (37, 34)), fill=TEAL), d.polygon(((17, 20), (29, 27), (17, 35)), fill=TEAL)))
