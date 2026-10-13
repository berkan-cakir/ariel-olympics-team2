"""
Race videos and a browsable gallery of a symmetric co-evolution run.

* :func:`race_video` -- the survivors of a generation racing side by side,
  fastest first, laid out as in the generation's picture
  (:func:`sc_render.save_grid`). Each robot is followed by its own camera at
  45 degrees, with a clock and a bar for the distance it has covered.
* :func:`write_gallery` -- ``<run folder>/seed_<NN>_gallery.html``: every
  generation with its race video, its fastest robot's video and its picture,
  newest first. It only links the files, so open it from the run folder.
* :func:`survivor_race_videos` and :func:`gallery_page` -- the hooks with
  which ``run_symmetric_coevolution.py`` keeps both up to date during a run;
  ``make_gallery.py`` makes them for generations that ran without them.

The robots are driven exactly as they were scored: ``settle`` seconds without
control, then the hinge commands of :func:`sc_core.servo_commands`, changed
every :data:`sc_core.CONTROL_STEPS` physics steps.
"""

# Standard library
import html
import json
import math
import operator
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import cv2
import mujoco
import numpy as np
from sc_core import (
    CONTROL_STEPS,
    EvalSettings,
    Gait,
    RunConfig,
    Snapshot,
    is_due,
    servo_commands,
)
from sc_render import COLUMNS, TILE
from view_best import (
    EXTRA_SECONDS,
    NamedRecorder,
    build_scene,
    put_text,
    race_clock,
    tracking_camera,
)

# Local libraries
from ariel.ec import Individual
from ariel.parameters.ariel_types import FloatArray
from ariel.utils.symmetry import find_core_body

# Global constants
HEADER = 44  # pixels above the tiles, for the title and the clock
RACE_FPS = 20  # 16 robots per frame: fewer frames, and robots move at 1 Hz
TILE_TEXT = 0.45  # font scale on a tile
CAMERA_DISTANCE = 1.3  # m; the "angle" view of view_best, closer
BAR = 5  # pixels: the bar for the distance covered
BACKGROUND = (20, 20, 22)
BAR_EMPTY = (70, 70, 74)
BAR_RUNNING = (42, 120, 214)  # the blue of the run's plots
BAR_FINISHED = (27, 175, 122)  # and its aqua
POSTER_AT = 0.4  # a video's poster is its frame this far in
POSTERS = "posters"  # folder of the posters, in the video folder
PAGE_STYLE = """
:root {
  --surface: #fcfcfb; --raised: #ffffff; --ink: #0b0b0b;
  --ink-2: #52514e; --ink-3: #7a7974; --line: #e4e3df; --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #141416; --raised: #1c1c1f; --ink: #f2f2f0;
    --ink-2: #b4b3ae; --ink-3: #8f8e89; --line: #2e2e33; --accent: #6aa5ea;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    sans-serif;
}
header, main { max-width: 1120px; margin: 0 auto; padding: 0 16px; }
header { padding-top: 28px; }
h1 { font-size: 22px; margin: 0 0 4px; }
p { margin: 0 0 4px; }
.lead { color: var(--ink-2); }
.small { color: var(--ink-3); font-size: 13px; }
nav { display: flex; flex-wrap: wrap; gap: 4px; margin: 14px 0 18px; }
nav a {
  min-width: 34px; padding: 1px 6px; border: 1px solid var(--line);
  border-radius: 6px; color: var(--ink-2); font-size: 12px;
  text-align: center; text-decoration: none;
}
nav a:hover, nav a:focus-visible {
  border-color: var(--accent); color: var(--accent);
}
details {
  margin: 0 0 10px; background: var(--raised);
  border: 1px solid var(--line); border-radius: 10px;
}
summary {
  display: flex; flex-wrap: wrap; gap: 2px 14px; align-items: baseline;
  padding: 10px 14px; cursor: pointer;
}
summary .gen { min-width: 118px; font-weight: 600; }
summary .stats { color: var(--ink-2); font-size: 14px; }
.media { padding: 2px 14px 14px; }
figure { margin: 0 0 14px; }
figcaption { margin-top: 4px; color: var(--ink-3); font-size: 13px; }
video, img {
  display: block; width: 100%; height: auto; border-radius: 6px;
  background: #000;
}
"""


@dataclass(frozen=True)
class Runner:
    """
    One robot of a race video.

    Parameters
    ----------
    rank
        Its place in the generation, from 1.
    uid
        Its uid.
    genome
        Its tree genome as a dict.
    gait
        Its learned gait (:meth:`sc_core.Gait.to_dict`), or ``None`` if it
        could not be simulated.
    finish_time
        When it crossed the finish line in its scored dash, or ``None``.
    """

    rank: int
    uid: int
    genome: dict[str, Any]
    gait: dict[str, Any] | None
    finish_time: float | None


def runner(rank: int, genome: dict[str, Any], tags: dict[str, Any]) -> Runner:
    """
    Make a runner from a stored body.

    Parameters
    ----------
    rank
        Its place in the generation, from 1.
    genome
        Its tree genome as a dict.
    tags
        Its tags (uid, gait, finish time).

    Returns
    -------
    Runner
        The robot to film.
    """
    finish = tags.get("finish_time")
    return Runner(
        rank=rank,
        uid=int(tags["uid"]),
        genome=genome,
        gait=tags.get("gait"),
        finish_time=None if finish is None else float(finish),
    )


class _Lane:
    """One robot's simulation, camera and renderer in a race video."""

    def __init__(
        self,
        robot: Runner,
        settings: EvalSettings,
        seconds: float,
    ) -> None:
        self.robot = robot
        self.settings = settings
        self.scene = build_scene(robot.genome, settings)
        model = self.scene.model
        period = CONTROL_STEPS * model.opt.timestep
        if robot.gait is None:
            self.commands = np.zeros((1, model.nu))
        else:
            self.commands = servo_commands(
                Gait.from_dict(robot.gait),
                settings,
                period,
                max(1, math.ceil(seconds / period)),
            )
        self.settle_steps = round(settings.settle / model.opt.timestep)
        self.steps = 0
        self.camera = tracking_camera(find_core_body(model), "angle")
        self.camera.distance = CAMERA_DISTANCE
        self.renderer = mujoco.Renderer(model, height=TILE[0], width=TILE[1])

    def advance(self, steps: int) -> None:
        """
        Step the physics on, as when the robot was scored.

        Parameters
        ----------
        steps
            Physics steps to take.
        """
        model, data = self.scene.model, self.scene.data
        target = self.steps + steps
        while self.steps < target:
            dash = self.steps - self.settle_steps
            if dash < 0:
                data.ctrl[:] = 0.0
                chunk = min(target, self.settle_steps) - self.steps
            else:
                update = min(dash // CONTROL_STEPS, len(self.commands) - 1)
                data.ctrl[:] = self.commands[update]
                chunk = min(
                    target - self.steps,
                    CONTROL_STEPS - dash % CONTROL_STEPS,
                )
            mujoco.mj_step(model, data, nstep=chunk)
            self.steps += chunk

    def tile(self) -> FloatArray:
        """
        Render the robot with its rank, clock and distance bar.

        Returns
        -------
        FloatArray
            An RGB image of :data:`sc_render.TILE` pixels.
        """
        data, robot, settings = self.scene.data, self.robot, self.settings
        self.renderer.update_scene(data, camera=self.camera)
        image: FloatArray = self.renderer.render()
        clock = race_clock(data, settings, self.scene.start, robot.finish_time)
        put_text(image, f"#{robot.rank}  uid {robot.uid}", (8, 20), TILE_TEXT)
        put_text(image, clock, (8, TILE[0] - 13), TILE_TEXT)
        dash = data.time - settings.settle
        done = robot.finish_time is not None and dash >= robot.finish_time
        covered = (data.qpos[0] - self.scene.start) / settings.goal
        filled = round(
            TILE[1] * (1.0 if done else float(np.clip(covered, 0, 1))),
        )
        image[-BAR:, :] = BAR_EMPTY
        image[-BAR:, :filled] = BAR_FINISHED if done else BAR_RUNNING
        return image

    def close(self) -> None:
        """Free the renderer."""
        self.renderer.close()


def race_video(
    robots: Sequence[Runner],
    settings: EvalSettings,
    path: Path,
    title: str,
) -> Path:
    """
    Film robots racing side by side, each followed by its own camera.

    The video shows the settling and the dash until the last robot finishes
    (plus one second) or the time limit.

    Parameters
    ----------
    robots
        The robots, in tile order: left to right, then top to bottom.
    settings
        The run's settings.
    path
        The video to write, ``<name>.mp4``.
    title
        Text above the tiles.

    Returns
    -------
    Path
        The video written; an ``.avi`` if OpenCV could write no MP4.
    """
    seconds = max(
        settings.time_limit
        if robot.finish_time is None
        else robot.finish_time + EXTRA_SECONDS
        for robot in robots
    )
    lanes = [_Lane(robot, settings, seconds) for robot in robots]
    try:
        timestep = lanes[0].scene.model.opt.timestep
        steps = max(1, round(1 / (RACE_FPS * timestep)))
        frames = math.ceil((settings.settle + seconds) / (steps * timestep))
        frame = _frame(lanes, steps, title)
        recorder = NamedRecorder(
            file_name=path.stem,
            output_folder=path.parent,
            width=frame.shape[1],
            height=frame.shape[0],
            fps=RACE_FPS,
        )
        recorder.write(frame)
        for _ in range(frames - 1):
            recorder.write(_frame(lanes, steps, title))
        recorder.release()
    finally:
        for lane in lanes:
            lane.close()
    return path if path.exists() else path.with_suffix(".avi")


def _frame(lanes: Sequence[_Lane], steps: int, title: str) -> FloatArray:
    rows = math.ceil(len(lanes) / COLUMNS)
    height, width = HEADER + rows * TILE[0], COLUMNS * TILE[1]
    frame = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    for index, lane in enumerate(lanes):
        lane.advance(steps)
        row, column = divmod(index, COLUMNS)
        top, left = HEADER + row * TILE[0], column * TILE[1]
        frame[top : top + TILE[0], left : left + TILE[1]] = lane.tile()
    dash = lanes[0].scene.data.time - lanes[0].settings.settle
    put_text(frame, title, (12, 29))
    clock = f"dash {dash:4.1f} s" if dash >= 0 else "settling"
    put_text(frame, clock, (width - 150, 29))
    return frame


def film_generation(  # noqa: PLR0913 -- a generation's video needs these
    robots: Sequence[Runner],
    settings: EvalSettings,
    video_dir: Path,
    generation: int,
    *,
    title: str,
    filmed: dict[tuple[int, ...], Path],
) -> Path:
    """
    Make a generation's race video, ``gen_<NNN>_all.mp4``.

    If the survivors and their order are those of a generation filmed
    before, its video is linked instead of filmed again.

    Parameters
    ----------
    robots
        The survivors, fastest first.
    settings
        The run's settings.
    video_dir
        The run's video folder.
    generation
        The generation.
    title
        Text above the tiles.
    filmed
        Survivors' uids to the video made of them; updated.

    Returns
    -------
    Path
        The generation's race video.
    """
    key = tuple(robot.uid for robot in robots)
    path = video_dir / f"gen_{generation:03d}_all.mp4"
    video_dir.mkdir(parents=True, exist_ok=True)
    if key in filmed:
        target = path.with_suffix(filmed[key].suffix)
        link(filmed[key], target)
        return target
    filmed[key] = race_video(robots, settings, path, title)
    return filmed[key]


def link(source: Path, target: Path) -> None:
    """
    Make ``target`` a hard link to ``source``, or a copy where links fail.

    Parameters
    ----------
    source
        An existing file.
    target
        The new name; replaced if it exists.
    """
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def race_title(seed: int) -> str:
    """
    Return the text above the tiles of a run's race videos.

    It names no generation, because a generation whose survivors did not
    change links the video of the one before.

    Parameters
    ----------
    seed
        The run's seed.

    Returns
    -------
    str
        E.g. ``seed 1 survivors, fastest first; cameras at 45 deg``.
    """
    return f"seed {seed} survivors, fastest first; cameras at 45 deg"


def survivor_race_videos(cfg: RunConfig) -> Snapshot:
    """
    Return the run's race-video hook: a generation's survivors, side by side.

    Each due generation (``cfg.video_every``, see :func:`sc_core.is_due`)
    gets ``<cfg.video_dir>/gen_<NNN>_all.mp4`` (:func:`film_generation`).

    Parameters
    ----------
    cfg
        Run settings (output folder, cadence, task and servo settings).

    Returns
    -------
    Snapshot
        A function of the survivors (best first) and the generation.
    """
    settings = cfg.eval_settings()
    filmed: dict[tuple[int, ...], Path] = {}

    def hook(survivors: list[Individual], generation: int) -> None:
        if not is_due(generation, cfg.video_every, cfg.gens):
            return
        robots = [
            runner(
                rank,
                cast("dict[str, Any]", ind.genotype)["morph"],
                cast("dict[str, Any]", ind.tags),
            )
            for rank, ind in enumerate(survivors, start=1)
        ]
        film_generation(
            robots,
            settings,
            cfg.video_dir,
            generation,
            title=race_title(cfg.seed),
            filmed=filmed,
        )

    return hook


# -- Gallery ------------------------------------------------------------------


def poster(video: Path) -> Path | None:
    """
    Return a still of a video for the gallery, saving it if needed.

    Parameters
    ----------
    video
        A video in a run's video folder.

    Returns
    -------
    Path | None
        ``<video folder>/posters/<name>.jpg``, or ``None`` if the video
        cannot be read.
    """
    still = video.parent / POSTERS / f"{video.stem}.jpg"
    if still.exists() and still.stat().st_mtime >= video.stat().st_mtime:
        return still
    capture = cv2.VideoCapture(str(video))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(POSTER_AT * frames)))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    still.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(still), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return still


def write_gallery(db_path: Path) -> Path:
    """
    Write a run's gallery page from its metadata and output files.

    Parameters
    ----------
    db_path
        The run database; the metadata, pictures and videos sit next to it.

    Returns
    -------
    Path
        ``<run folder>/seed_<NN>_gallery.html``.
    """
    run_dir, stem = db_path.parent, db_path.stem
    meta = json.loads((run_dir / f"{stem}_meta.json").read_text("utf-8"))
    newest_first = list(reversed(meta["history"]))
    sections = [
        _generation_section(
            run_dir,
            stem,
            entry,
            meta["config"],
            newest=entry is newest_first[0],
        )
        for entry in newest_first
    ]
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, '
        'initial-scale=1">\n'
        f"<title>{html.escape(run_dir.name)} {stem} gallery</title>\n"
        f"<style>{PAGE_STYLE}</style>\n</head>\n<body>\n"
        f"{_page_header(run_dir.name, stem, meta)}\n"
        f"<main>\n{''.join(sections)}</main>\n</body>\n</html>\n"
    )
    path = run_dir / f"{stem}_gallery.html"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(path)
    return path


def _page_header(name: str, stem: str, meta: dict[str, Any]) -> str:
    cfg, history = meta["config"], meta["history"]
    last = history[-1]["generation"] if history else 0
    best = max(history, key=operator.itemgetter("best_speed"), default=None)
    fastest = "" if best is None else f" Fastest so far: {_result(best, cfg)}."
    gain = cfg.get("servo_gain", 0)
    servos = (
        f"servos {cfg.get('max_hinge_speed', 0):g} rad/s, "
        f"{cfg.get('max_hinge_torque', 0):g} N m, "
        f"{cfg.get('max_hinge_frequency', 0):g} Hz, "
        + (f"gain {gain:g} N m/rad" if gain > 0 else "ARIEL's slow servo")
    )
    settings = (
        f"{cfg['brain']} brain; {cfg['pop']} + {cfg['offspring']} bodies per "
        f"generation; {cfg['learn_gens'] * cfg['learn_pop']} races to learn "
        f"each gait; {servos}; pruning {'on' if cfg.get('prune') else 'off'};"
        f" grow {cfg.get('grow', 0):g}; double servos "
        f"{cfg.get('double_servo', 0):g}"
    )
    links = "".join(
        f'<a href="#gen-{entry["generation"]:03d}">{entry["generation"]}</a>'
        for entry in history
    )
    return (
        f"<header>\n<h1>{html.escape(name)}, {html.escape(stem)}</h1>\n"
        f'<p class="lead">Generation {last} of {cfg["gens"]} '
        f"({html.escape(meta.get('status', ''))}).{fastest}</p>\n"
        f'<p class="small">{html.escape(settings)}</p>\n'
        f'<p class="small">Page written {time.strftime("%Y-%m-%d %H:%M")};'
        " reload it to see newer generations. Videos load when played.</p>\n"
        f'<nav aria-label="Generations">{links}</nav>\n</header>'
    )


def _generation_section(
    run_dir: Path,
    stem: str,
    entry: dict[str, Any],
    cfg: dict[str, Any],
    *,
    newest: bool,
) -> str:
    generation = entry["generation"]
    videos = run_dir / f"{stem}_videos"
    race = _first(videos, f"gen_{generation:03d}_all.*")
    fastest = _first(videos, f"gen_{generation:03d}_uid*.*")
    picture = run_dir / f"{stem}_snapshots" / f"gen_{generation:03d}.png"
    previous = _first(videos, f"gen_{generation - 1:03d}_all.*")
    parts = []
    if race is not None:
        same = previous is not None and Path(race).samefile(previous)
        note = (
            f" Same survivors as generation {generation - 1}." if same else ""
        )
        parts.append(
            _video(
                run_dir,
                race,
                "The survivors racing, fastest first, each followed by its "
                f"own camera at 45 degrees.{note}",
            ),
        )
    if fastest is not None:
        parts.append(
            _video(
                run_dir,
                fastest,
                f"The fastest, uid {entry['best_uid']}, from the top, the "
                "side and 45 degrees.",
            ),
        )
    if picture.exists():
        source = html.escape(picture.relative_to(run_dir).as_posix())
        parts.append(
            f'<figure><a href="{source}"><img loading="lazy" src="{source}" '
            f'alt="The survivors of generation {generation} at their '
            'starting pose"></a><figcaption>The survivors at their starting '
            "pose, fastest first (click to enlarge).</figcaption></figure>",
        )
    if not parts:
        parts.append('<p class="small">No pictures or videos yet.</p>')
    is_open = " open" if newest else ""
    stats = (
        f"fastest {_result(entry, cfg)}; average {entry['mean_speed']:.3f} "
        f"m/s; {entry['finishers']} finish; "
        f"{entry.get('mean_modules', 0):.1f} modules on average"
    )
    return (
        f'<details id="gen-{generation:03d}"{is_open}>\n<summary>'
        f'<span class="gen">Generation {generation}</span>'
        f'<span class="stats">{html.escape(stats)}</span></summary>\n'
        f'<div class="media">\n{"".join(parts)}\n</div>\n</details>\n'
    )


def _video(run_dir: Path, video: Path, caption: str) -> str:
    source = html.escape(video.relative_to(run_dir).as_posix())
    still = poster(video)
    attribute = (
        ""
        if still is None
        else f' poster="{html.escape(still.relative_to(run_dir).as_posix())}"'
    )
    return (
        f'<figure><video controls preload="none"{attribute} src="{source}">'
        f"</video><figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _result(entry: dict[str, Any], cfg: dict[str, Any]) -> str:
    finish = entry.get("best_finish_time")
    reach = (
        "did not finish"
        if finish is None
        else f"{cfg['goal']:g} m in {float(finish):.1f} s"
    )
    return (
        f"uid {entry['best_uid']}, {reach} ({entry['best_speed']:.3f} m/s, "
        f"{entry.get('best_family') or 'no'} gait)"
    )


def _first(folder: Path, pattern: str) -> Path | None:
    videos = sorted(
        path for path in folder.glob(pattern) if path.suffix in {".mp4", ".avi"}
    )
    return videos[0] if videos else None


def gallery_page(cfg: RunConfig) -> Snapshot:
    """
    Return the run's gallery hook, which rewrites the page every generation.

    Use it after the picture and video hooks, so that it links their files.

    Parameters
    ----------
    cfg
        Run settings (output folder).

    Returns
    -------
    Snapshot
        A function of the survivors and the generation.
    """

    def hook(survivors: list[Individual], generation: int) -> None:
        del survivors, generation
        write_gallery(cfg.db_path)

    return hook
