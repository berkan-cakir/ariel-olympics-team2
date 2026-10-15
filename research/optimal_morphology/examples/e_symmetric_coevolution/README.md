# Symmetric body–brain co-evolution: a 3 m dash

Evolve mirror-symmetric modular robots, and their controllers, to move **3 m
forward within 60 s, as fast as possible**. Every new body learns its own
controller from scratch before it is judged.

## The task

After 2 s of settling without control, the robot has 60 s to move its core 3 m
forward (+x, the direction it faces at spawn). Its score is its average forward
speed over the dash:

| Outcome | Score |
|---|---|
| Crosses the line after *t* seconds | 3 m / *t* (3 m in 15 s scores 0.2 m/s) |
| Does not reach the line in 60 s | furthest distance / 60 s |

The two cases meet at the line (0.05 m/s), so every finisher beats every
non-finisher, and faster finishers beat slower ones. A race stops as soon as
the robot finishes, so fast robots are also cheap to simulate.

## Realistic servos

ARIEL's simulated hinge has a position servo with no limit on speed or
torque. Without limits, evolution found robots that buzz: the two runs' winners
commanded ±88° swings at 5.5–6.2 Hz. The hinge could not follow (it moved
about ±5°) while the actuator reversed at full effort a dozen times a second
with up to 3.9 N m. A real servo turns at a bounded, roughly constant speed,
has a torque limit, and overheats if switched back and forth that fast. So
every hinge here is limited like a real servo:

| Setting | Default | Effect |
|---|---|---|
| `--max-hinge-speed` | 5.24 rad/s (300°/s: 60° in 0.2 s) | The commanded angle moves toward the gait's target at most this fast, so the servo is never sent a command it could not follow. |
| `--max-hinge-torque` | 1.0 N m | The actuator torque is clamped (MuJoCo `forcerange`). |
| `--max-hinge-frequency` | 1.0 Hz | No hinge oscillates faster, i.e. at most 2 direction changes per second. This also counts hinges that run at twice the gait's frequency. |

The defaults are a standard hobby servo under load: the hinge housing is 5 cm
across and the core weighs 1 kg, and lifting about 1.4 kg on 10 cm legs takes
about 0.7 N m per pair of legs. **Set them from your servo's datasheet.** Use
the speed under load, the torque you can sustain rather than the stall
torque, and the switching rate the servo tolerates. Each limit can be turned
off with 0, and runs with any limit on go to `<brain>[_pruned]_servo/`.

Every scored body also gets its dash measured, stored in its tags and in
`plot_run.py`'s bodies table:

- `peak_torque` and `rms_torque` (N m). The RMS torque of the busiest hinge is
  the heating measure, since motor losses grow with the square of the current,
  i.e. of the torque. Compare it with the servo's continuous rating.
- `peak_hinge_speed` (rad/s).
- `reversals_per_s`: direction changes per second of the busiest hinge,
  counting only swings of at least 5°, so contact jitter is ignored.
- `hinge_frequency` (Hz).

Under these limits the old winners slow down sharply. With newly learned
gaits, run 1's winner needed 55–57 s for the 3 m instead of 5.2 s, and the
pruned run's winner about 49 s instead of 5.5 s. Their reversal rates dropped
to 1.4–1.7 per second, as the 1 Hz cap allows. Runs made before the limits
existed replay exactly as they were scored, because their metadata has no
limits.

### A hobby servo instead of ARIEL's

The limits above sit on top of ARIEL's servo, which pushes 1 N m per radian
off target and brakes 1 N m per rad/s of speed. That makes a hinge lag about a
second behind its command. The fastest robot of the 100-generation run with
growth (uid 312) moved its hinges at 0.88 Hz and swung them only 16–24° of the
124–143° commanded, and never reached the 1 N m limit. So the limits hardly
mattered: ARIEL's sluggish servo was the real limit.

`--servo-gain` (default 10 N m/rad) makes every hinge act like a hobby servo
(`set_servos` in `sc_core.py`):

- it pushes with full torque from about 6° off target;
- like a motor, its torque falls with speed, to 0 at `--max-hinge-speed`
  (joint damping of torque / speed, 0.19 N m s/rad by default);
- it has a motor's inertia through the gearbox, 0.001 kg m² (an estimate).

With the same gait, uid 312's hinges then swing 114–131° and sit at the 1 N m
limit 82–86% of the time. Runs with this servo go to
`<brain>[_pruned]_realservo[_grow]/`; `--servo-gain 0` keeps ARIEL's servo, as
in the runs made before.

### Two servos per hinge

A body is a tree in which every module has one parent, and a hinge has one
servo. Nothing in the encoding could put two servos on one joint, and two
hinges in series do not add torque: each still carries the whole load. A
servo switch (`toggle_servos` in `sc_core.py`) changes a random hinge
between one servo and two in parallel, and mirroring copies the switch to the
hinge's twin. Two servos that get the same command add their torques. So the
hinge gets twice the gain, torque limit, damping and motor inertia, and the
second servo's 60 g goes into the hinge's housing. The two servos sit side by
side on the hinge's axis, so the hinge is twice as wide along it: 10 cm, a
block's width. It still fits wherever a hinge fits: none of the 1107 bodies of
a realistic-servo run overlapped itself with all its hinges doubled. Such
hinges are drawn purple.

`--double-servo` is the chance that a mutated child also switches one of its
hinges (default 0.2). The switch comes on top of the child's other mutation,
so the other mutations keep exactly their rates and a run with double servos
differs from one without only in the switches. Runs with it go to a folder
ending in `_double`. Each body's tags count its double hinges (`n_double`), and
the torques measured per dash are per servo. The only price of a second servo
is its weight; power use and cost are not part of the score.

## Why the controller is learned for every body

Optimising body and controller together in one genome tends to get stuck. A
mutated body inherits a controller made for its parent's body, so it usually
performs worse than it could, and evolution keeps the first body that works.
Here the two are optimised in nested loops instead:

- **Evolution (outer loop).** A (μ + λ) evolutionary algorithm over tree genomes
  with tournament parent selection, subtree crossover and the tree mutations of
  `examples/d_design_priors`. Every body is mirror-symmetric about the robot's
  sagittal plane, in module positions *and* hinge axes
  (`ariel.ec.genotypes.tree.symmetry.symmetrize`), and this is checked on the
  compiled MuJoCo body. Every offspring is a body that is not already in the
  population.
- **Learning (inner loop).** Every new body gets a controller learned from
  scratch by CMA-ES, with the same budget for every body (`learn_gens ×
  learn_pop` races, 192 by default). A body's fitness is the speed of the best
  controller found for it.

## Pruning: no legs that never touch the ground

Random bodies often carry legs and hinges that the learned gait never puts on
the ground. By default (`--prune`), every new body is simplified after it has
learned its gait:

1. its best dash is replayed and every module that touches the floor is
   recorded, from the start until the finish line (or 60 s);
2. every subtree (a leg, or the end of one) in which no module touched the
   floor is cut off. A module that never touches the floor stays if something
   below it does, and mirror twins are judged together, so the body stays
   symmetric;
3. the pruned body learns its own gait with the same budget, and it replaces
   the original in the population, with the faster of that gait and the old
   gait of its remaining hinges.

The size and speed before pruning stay in the tags (`unpruned_modules`,
`unpruned_hinges`, `unpruned_speed`, `cut`). Two bodies that prune to the same
body are not both kept. A pruned body is learned twice, but most offspring of
pruned parents need no pruning, and small bodies simulate faster.
`--no-prune` turns it off (the run folder's name then drops `_pruned`).

Pruning is strict. A leg that only swings in the air, as a counterweight, is
removed too, and a robot whose gait never puts any leg down (it shuffles on its
core) loses all of them; such bodies score poorly and die out. In the first run's fastest
starting robot (uid 12), only 4 of 20 modules ever touched the floor. Pruning
left 9 modules and 5 hinges, and with gaits learned from scratch (3 seeds each)
the pruned body averaged 0.199 m/s against 0.209 m/s for the original. With
the original's old gait it was much slower (3 m in 29.3 s instead of 18.5 s):
the cut legs had been swinging as counterweights.

`prune_body.py` prunes one stored body on its own and compares the two bodies;
`view_best.py` can show the result:

```bash
uv run python examples/e_symmetric_coevolution/prune_body.py \
    ~/ariel_runs/symmetric_coevolution/symmetric/seed_01.db --uid 12 --video ~/ariel_runs/videos
uv run python examples/e_symmetric_coevolution/view_best.py \
    ~/ariel_runs/symmetric_coevolution/symmetric/pruned/symmetric_seed_01_uid12.json
```

## Growing limbs

Every module is 10 cm long, so a limb only gets longer by gaining a module. The
tree mutations of `examples/d_design_priors` rarely do that. In the first 50
generations of the servo run without growth, 4% of children had a longer
longest limb than either parent and 27% a shorter one, and pruning only cuts.

The `grow` mutation (`grow_limb` in `sc_core.py`) puts a new brick or hinge at a
random place along a limb. It goes either between a module and its parent, which
pushes the rest of the limb one module outward, or after the limb's last module.
Mirror symmetry then copies it to the twin limb.

`--grow` is the share of mutations that grow (default 0.4); the other mutations
keep their proportions. At 0.4, children of that run's populations came out
longer about as often as shorter (20–23% against 23%), compared with 2–3%
against 33–38% without growth. Runs with growth go to a folder ending in
`_grow`. `--grow 0` runs, or resumes, the experiment without growth. Each
body's `mutation` tag says which mutation made it.

## The controller: a symmetric central pattern generator

Each hinge follows a sine wave, all at one learned frequency (an open-loop
central pattern generator). The controller is built from the body: its hinges
are matched to their mirror images on the compiled model, and the waves are tied
together by a *gait family*:

| Family | How the hinge waves are tied |
|---|---|
| alternating | Mirror pairs share amplitude and offset and run half a period apart. At *t + T/2* the robot is the mirror image of itself at *t*, as in a walk or trot. |
| synchronous | Mirror pairs are mirror images at every instant, as in a bound. |
| free | Every hinge has its own wave (no symmetry). |

A symmetric family makes a symmetric body move straight ahead or straight back
without having to learn to steer. Over 60 s, 96 random gaits per family on 12
symmetric bodies turned as follows:

| Family | Median turn | 90th percentile turn | 90th percentile sideways drift |
|---|---|---|---|
| alternating | 0.4° | 4° | 1.9 cm |
| synchronous | 0.0° | 0.0° (max 0.2°) | 0.0 cm |
| free | 9° | 75° | 25 cm |

The worst alternating gait still turned by about 50°: a robot can settle into a
lopsided version of its gait. Easing the gait in over 1–2 s did not prevent
this.

**Which symmetric family suits a body depends on the body.** In a pilot, 12
symmetric bodies (4 hand-designed, 8 random) each learned every family twice
with 288 races:

| Family | Median speed | Mean speed | Finished 3 m | Median turn |
|---|---|---|---|---|
| alternating | 0.027 m/s | 0.033 m/s | 3 of 24 | 2.3° |
| synchronous | 0.006 m/s | 0.045 m/s | 8 of 24 | 0.0° |
| free | 0.023 m/s | 0.025 m/s | 3 of 24 | 41.5° |

Synchronous gaits made the fastest robots (one body covered 3 m in about 11 s)
but left most bodies barely moving. Alternating gaits were more reliable, beating
synchronous ones on 8 of the 12 bodies. So the default `--brain symmetric` learns
**both** symmetric families, with half the budget each, and keeps the faster
gait. Estimated from the pilot's learning curves, splitting its 288 races this
way gives a mean of 0.040 m/s and a median of 0.020 m/s: close to the best
single family on both counts. Each body's chosen family is stored and plotted.

| `--brain` | Families learned |
|---|---|
| `symmetric` (default) | alternating and synchronous, half the budget each |
| `alternating`, `synchronous`, `free` | that family only, e.g. for a control run |

## Running it

```bash
# One run (seed 1) on all but one core, 2-3 h on an M2; keeps the Mac awake.
# Add e.g. --max-hinge-torque 1.5 to match your servos.
caffeinate -dims uv run python \
    examples/e_symmetric_coevolution/run_symmetric_coevolution.py --seed 1

# Plots: progress, the best body over time, and CSVs of every generation/body.
uv run python examples/e_symmetric_coevolution/plot_run.py \
    ~/ariel_runs/symmetric_coevolution/symmetric_pruned_realservo_grow_double/seed_01.db

# Watch the fastest robot in MuJoCo's viewer (Space pauses, Backspace
# restarts), or film it from the top, the side and 45 degrees.
uv run python examples/e_symmetric_coevolution/view_best.py \
    ~/ariel_runs/symmetric_coevolution/symmetric_pruned_realservo_grow_double/seed_01.db
uv run python examples/e_symmetric_coevolution/view_best.py \
    ~/ariel_runs/symmetric_coevolution/symmetric_pruned_realservo_grow_double/seed_01.db --video ~/ariel_runs/videos
```

`--help` lists every setting. A quick smoke test (about 15 s):

```bash
uv run python examples/e_symmetric_coevolution/run_symmetric_coevolution.py \
    --seed 1 --out /tmp/smoke --pop 4 --offspring 3 --gens 2 \
    --learn-gens 2 --learn-pop 4 --time-limit 5 --goal 0.3 --video-every 0
```

A run that stopped, or one that should go on for more generations, continues
with `--resume` and the new total of `--gens`:

```bash
caffeinate -dims uv run python \
    examples/e_symmetric_coevolution/run_symmetric_coevolution.py --seed 1 \
    --gens 100 --resume
```

The resumed run takes the survivors of the last finished generation from the
database. To bring the random state back, it redoes the variation of every
earlier generation, which takes seconds because nothing is learned, and checks
that it makes the stored bodies. The run then goes on exactly as if it had never
stopped. The settings must be the run's, apart from `--gens`, `--workers`,
`--snapshot-every`, `--video-every`, `--reeval-top` and `--reeval-seeds`. The
database, metadata and re-evaluation as they were are first copied to
`seed_NN_backup_genNNN/`. With `--gens` equal to the last generation, only the
final re-evaluation is redone.

For several seeds, run them one after the other (each run already uses all
cores), then plot them together:

```bash
for seed in 1 2 3 4 5; do
    uv run python examples/e_symmetric_coevolution/run_symmetric_coevolution.py --seed $seed
done
uv run python examples/e_symmetric_coevolution/plot_run.py \
    ~/ariel_runs/symmetric_coevolution/symmetric_pruned_realservo_grow_double/seed_0*.db
```

The run time scales with the number of bodies, `pop + gens × offspring` (336 by
default), times the races per body. A 60 s race takes about 0.76 s on one M2 core
when the robot does not finish, so a body costs about 2.5 CPU-minutes; fast
finishers cost less. Without servo limits, seed 1 took 2 h 2 min (1 h 47 min
with pruning) on an M2 with 7 workers. With the limits, robots are slower and
more races run the full 60 s: seed 1 took 4 h 41 min for 50 generations.

## Outputs

Runs go to
`~/ariel_runs/symmetric_coevolution/<brain>_pruned_realservo_grow_double/`. The
folder name drops `_pruned` with `--no-prune`, `_grow` with `--grow 0` and
`_double` with `--double-servo 0`; with `--servo-gain 0`, `_realservo` becomes
`_servo`, or nothing when all servo limits are 0 too. Keep them out of
OneDrive, where syncing can corrupt SQLite files.

- `seed_NN.db`: the EA database. Every body has its genome (pruned, if pruning
  cut anything) and tags: origin, parents, the mutation that made it, size,
  mirror pairs, measured symmetry, speed, finish time, drift and turn, gait
  family, the speed each family reached, the learned gait and its learning
  curve, and the size and
  speed before pruning.
- `seed_NN_meta.json`: settings, versions, status, and one line per generation.
  It is rewritten every generation, so it also shows a running run's progress.
- `seed_NN_reeval.json`: the 3 fastest bodies, each re-learned with 3 fresh
  seeds. A body's run speed is the best of one learning run, so the survivors
  are biased toward lucky learning runs; the re-learned mean is the fairer
  number to report.
- `seed_NN_snapshots/gen_NNN.png`: **screenshots of the morphologies** every
  `--snapshot-every` generations (default 1, i.e. every generation), plus the
  last. Each shows the surviving bodies, fastest first, with speed, time to 3
  m, gait family, size (and the size before pruning) and uid (`view_best.py
  --uid` shows any of them).
- `seed_NN_videos/gen_NNN_uidN.mp4`: **a video of each generation's fastest
  robot** (`--video-every`, default 1), filmed from the top, the side and 45°
  with a race clock. A robot that stays fastest is filmed once; later
  generations get a hard link to the same file.
- `seed_NN_videos/gen_NNN_all.mp4`: **all survivors racing side by side**,
  fastest first, laid out as in the screenshot. Each robot is followed by its
  own camera at 45°, with its time, distance and a bar toward the finish. A
  generation whose survivors did not change links the video of the one
  before. Videos are H.264, so browsers play them; `posters/` holds a still
  of each one.
- `seed_NN_gallery.html`: **every generation on one page**, newest first, with
  its race video, its fastest robot's video and its screenshot. The run
  rewrites it after every generation; open it from the run folder (`open
  seed_01_gallery.html`) and reload it for newer generations. For runs made
  before it existed, `make_gallery.py <db>` films the missing race videos,
  converts older MPEG-4 Part 2 videos to H.264 and writes the page.
- `seed_NN_backup_genNNN/`: the database, metadata and re-evaluation as they
  were before `--resume` continued the run after generation NNN.
- `plots/` (from `plot_run.py`):
  - `<run>_progress.png`: speed of the fastest and average survivor, every
    body's speed by birth, survivors that finish, their gait families, size of
    the fastest body, and learning curves;
  - `<run>_best_bodies.png`: the fastest body at up to 8 generations;
  - `<run>_generations.csv` and `<run>_bodies.csv`: the data behind the figures;
  - `runs_progress.png`: written when several runs are plotted together.

## Notes

- **Reproducibility:** the same seed gives the same run with any number of
  workers. `view_best.py` recomputes the stored speed from the stored gait
  before it shows anything, and the viewer replays exactly what was scored.
- **Resumable:** `--resume` continues a run as if it had never stopped. A small
  run stopped after 2 generations and resumed to 4 matched the same run made in
  one go in every body, fitness and tag. A new run never overwrites a database:
  it stops at once if one exists.
- **Open loop:** the controller has no sensors. That keeps learning cheap (7–34
  parameters for the pilot's bodies) and suits flat ground, but it cannot
  correct a stumble or a drift; rough terrain would need a closed-loop brain.
- **Files:**
  - `sc_core.py`: the experiment (task, controller, learning, pruning, bodies,
    EA);
  - `sc_render.py`: draws the bodies;
  - `sc_gallery.py`: the race videos and the gallery page;
  - `run_symmetric_coevolution.py`, `view_best.py`, `plot_run.py`,
    `prune_body.py`, `make_gallery.py`: the scripts.
