import argparse
import time

from gecko_hardware import GeckoHardware, ServoMapping
from run_gecko import sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("channels", type=int, nargs="+")
    parser.add_argument("--readback", type=int, nargs="+", default=None)
    parser.add_argument("--neutral", type=float, nargs="+", default=None)
    parser.add_argument("--angle", type=float, default=30.0)
    parser.add_argument("--no-readback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = len(args.channels)
    readback = args.readback or [None] * n
    neutral = args.neutral or [90.0] * n
    if len(readback) != n or len(neutral) != n:
        raise SystemExit("--readback and --neutral need one value per channel")
    mappings = [ServoMapping(f"channel {c}", c, r, d) for c, r, d in zip(args.channels, readback, neutral, strict=True)]
    robot = GeckoHardware(mappings)
    try:
        robot.neutral()
        time.sleep(1.0)
        sweep(robot, args.angle, not args.no_readback)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
