"""
Make the race videos and the gallery page of a symmetric co-evolution run.

Example
-------
    uv run python examples/e_symmetric_coevolution/make_gallery.py \
        ~/ariel_runs/symmetric_coevolution/symmetric_pruned_servo_grow/seed_01.db

A run makes both as it goes (``run_symmetric_coevolution.py``). This script
makes them for the generations that ran without them: it films every
generation's survivors racing side by side (:func:`sc_gallery.race_video`)
unless that video exists, converts the run's videos written as MPEG-4 Part 2,
which browsers do not play, to H.264, and writes ``seed_<NN>_gallery.html``
next to the database. It can run while the run goes on.
"""

# Standard library
import argparse
import json
import os
from pathlib import Path
from typing import Any

# Third-party libraries
import cv2
from sc_core import load_individuals
from sc_gallery import (
    Runner,
    film_generation,
    race_title,
    runner,
    write_gallery,
)
from view_best import run_settings

# Global constants
MPEG4_PART2 = frozenset({"mp4v", "fmp4"})  # codec tags, lower case


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns
    -------
    argparse.Namespace
        The run database and whether to film missing race videos.
    """
    parser = argparse.ArgumentParser(
        description="Make the race videos and gallery page of a run.",
    )
    parser.add_argument("db", type=Path, help="a run database")
    parser.add_argument(
        "--no-race-videos",
        dest="race_videos",
        action="store_false",
        help="only convert videos and write the page",
    )
    return parser.parse_args()


def survivors_of(
    rows: list[dict[str, Any]],
    generation: int,
    last: int,
) -> list[Runner]:
    """
    Return the survivors of a generation, fastest first, from the database.

    Parameters
    ----------
    rows
        Every individual, from :func:`sc_core.load_individuals`.
    generation
        The generation.
    last
        The last generation in the database.

    Returns
    -------
    list[Runner]
        The bodies kept that generation, ranked as in the run.
    """
    # A body kept in a generation is in the next one, so it dies later; in
    # the last generation, the bodies cut also die then, so ask "alive".
    kept = [
        row
        for row in rows
        if row["birth"] <= generation
        and (
            row["death"] > generation
            or (row["death"] == generation == last and row["alive"])
        )
    ]
    kept.sort(
        key=lambda row: (row["fitness"], row["tags"]["uid"]),
        reverse=True,
    )
    return [
        runner(rank, row["genotype"]["morph"], row["tags"])
        for rank, row in enumerate(kept, start=1)
    ]


def fourcc(video: Path) -> str:
    """
    Return the codec tag of a video.

    Parameters
    ----------
    video
        A video file.

    Returns
    -------
    str
        As OpenCV reports it, e.g. ``h264`` for H.264 and ``FMP4`` for
        MPEG-4 Part 2 (written with the tag ``mp4v``).
    """
    capture = cv2.VideoCapture(str(video))
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    return code.to_bytes(4, "little").decode("ascii", errors="replace")


def reencode(source: Path, target: Path) -> bool:
    """
    Copy a video as H.264.

    Parameters
    ----------
    source
        The video to convert.
    target
        The H.264 copy, an ``.mp4``.

    Returns
    -------
    bool
        Whether it worked; OpenCV may have no H.264 encoder.
    """
    capture = cv2.VideoCapture(str(source))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"avc1"),
        capture.get(cv2.CAP_PROP_FPS),
        size,
    )
    if not writer.isOpened():
        capture.release()
        return False
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    capture.release()
    return True


def convert_to_h264(video_dir: Path) -> int:
    """
    Convert a run's MPEG-4 Part 2 videos to H.264, in place.

    Hard links to one video (a robot that stayed fastest) stay links to
    one converted video.

    Parameters
    ----------
    video_dir
        The run's video folder.

    Returns
    -------
    int
        Videos converted (links not counted).
    """
    converted: dict[tuple[int, int], Path] = {}
    count = 0
    for video in sorted(video_dir.glob("*.mp4")):
        # Only known MPEG-4 Part 2 tags: a video still being written by the
        # run reads as something else and must be left alone.
        if fourcc(video).lower() not in MPEG4_PART2:
            continue
        status = video.stat()
        key = (status.st_dev, status.st_ino)
        temporary = video.with_name(f".{video.stem}.h264.mp4")
        temporary.unlink(missing_ok=True)
        if key in converted:
            os.link(converted[key], temporary)
        elif reencode(video, temporary):
            count += 1
        else:
            temporary.unlink(missing_ok=True)
            continue
        temporary.replace(video)
        converted[key] = video
    return count


def main() -> None:
    """Film the missing race videos and write the gallery page."""
    args = parse_args()
    db: Path = args.db
    video_dir = db.parent / f"{db.stem}_videos"
    if video_dir.exists():
        count = convert_to_h264(video_dir)
        print(f"Converted {count} videos to H.264", flush=True)  # noqa: T201
    if args.race_videos:
        film_missing(db, video_dir)
    print(f"Gallery: {write_gallery(db)}")  # noqa: T201


def film_missing(db: Path, video_dir: Path) -> None:
    """
    Film the race video of every generation in the database that has none.

    Parameters
    ----------
    db
        The run database.
    video_dir
        The run's video folder.
    """
    settings = run_settings(db)
    meta = json.loads(db.with_name(f"{db.stem}_meta.json").read_text())
    title = race_title(meta["config"]["seed"])
    rows = load_individuals(db)
    last = max(row["death"] for row in rows)
    filmed: dict[tuple[int, ...], Path] = {}
    for generation in range(last + 1):
        robots = survivors_of(rows, generation, last)
        video = video_dir / f"gen_{generation:03d}_all.mp4"
        if video.exists():
            filmed[tuple(robot.uid for robot in robots)] = video
            continue
        made = film_generation(
            robots,
            settings,
            video_dir,
            generation,
            title=title,
            filmed=filmed,
        )
        print(f"Generation {generation}: {made.name}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
