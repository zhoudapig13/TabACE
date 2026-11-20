"""Redraw the Figure 1 context spine as one uniform top-layer vector stroke."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


SPINE_X = 247.2
SPINE_Y_MIN = 95.04
SPINE_Y_MAX = 361.44
# Match the four horizontal context branches in the approved Figure 1.
SPINE_WIDTH = 1.18
SPINE_RGB = (37 / 255, 37 / 255, 37 / 255)


def redraw_spine(source: Path, destination: Path) -> None:
    reader = PdfReader(source)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected a one-page figure PDF, found {len(reader.pages)} pages")

    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    overlay_buffer = io.BytesIO()
    overlay = canvas.Canvas(overlay_buffer, pagesize=(width, height), pageCompression=1)
    overlay.setStrokeColorRGB(*SPINE_RGB)
    overlay.setLineWidth(SPINE_WIDTH)
    overlay.setLineCap(0)  # Butt caps meet the top and bottom branches exactly.
    overlay.line(SPINE_X, SPINE_Y_MIN, SPINE_X, SPINE_Y_MAX)
    overlay.save()
    overlay_buffer.seek(0)

    page.merge_page(PdfReader(overlay_buffer).pages[0], over=True)

    writer = PdfWriter()
    writer.add_page(page)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        writer.write(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    redraw_spine(args.source, args.destination)


if __name__ == "__main__":
    main()
