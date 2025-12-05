"""
Build an animated GIF from the rendered PNG frames in video_stuff/frames.
Assumes directories are named frame_0, frame_1, ..., each containing frame_<i>.png.
"""

from pathlib import Path
from typing import List

from PIL import Image

FPS = 15
OUTPUT = Path("video_stuff") / "collision_detection.gif"
FRAMES_DIR = Path("video_stuff") / "frames"


def load_frame_paths() -> List[Path]:
    """Return ordered list of frame PNGs based on numeric suffix."""
    frame_paths: List[Path] = []
    for frame_dir in sorted(FRAMES_DIR.glob("frame_*"), key=lambda p: int(p.name.split("_")[1])):
        png_path = frame_dir / f"{frame_dir.name}.png"
        if png_path.exists():
            frame_paths.append(png_path)
    return frame_paths


def build_gif(paths: List[Path]) -> None:
    """Create the GIF at OUTPUT using the provided frame paths."""
    if not paths:
        raise ValueError("No frames found to build the GIF.")
    images = [Image.open(p).convert("RGBA") for p in paths]
    duration_ms = int(1000 / FPS)
    images[0].save(
        OUTPUT,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def main() -> None:
    frame_paths = load_frame_paths()
    print(f"Found {len(frame_paths)} frames")
    build_gif(frame_paths)
    print(f"GIF written to {OUTPUT}")


if __name__ == "__main__":
    main()
