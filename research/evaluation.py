import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--environments", type=int, nargs="+", choices=[0, 1, 2, 3], default=None)
    p.add_argument("--video", type=Path, default=None, help="output folder; enables recording")
    return p.parse_args()


def record_video(checkpoint: Path, env_id: int, out_dir: Path) -> None:
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
