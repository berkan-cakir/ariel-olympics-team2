# Design priors in body–brain co-optimization

Liu, Guo & Cangelosi (2025, *ACM Computing Surveys*, §3.5.4) name a future direction for body–brain co-optimization: add design know-how, physical constraints and outside knowledge to make it more efficient. This experiment tests three such priors, each on its own, against a baseline.

| Condition | Prior |
|---|---|
| `baseline` | Random initial bodies (8–20 modules). |
| `seeded` | Half the initial population is ARIEL's hand-designed robots: spider, gecko, baby_a, baby_b, snake, linkin_modified, spider_8, turtle. |
| `symmetric` | Every body is mirror-symmetric about the robot's own sagittal plane, in module positions and hinge axes. |
| `joint_budget` | At most 8 hinges; extra hinges become bricks. |

Settings shared by every condition:
- Bodies have at most 20 modules, depth at most 12, and at least 1 hinge.
- Robots spawn facing the target.
- Each body's MLP brain is learned from scratch by CMA-ES at a fixed cost: `learn_budget` × `learn_pop` simulations.
- The EA is (μ+λ) with truncation selection.

## Running it

```bash
# 1. Pilot (~35 min on an M2): measures speed, picks --gens for the time
#    budget, and fixes the efficiency threshold tau_fit.
uv run python examples/d_design_priors/pilot.py --hours 8.5 --jobs 7

# 2. The batch: 4 conditions x 10 seeds, 7 at a time. It uses the pilot's
#    budget, and can be stopped and restarted (finished runs are kept).
caffeinate -dims uv run python examples/d_design_priors/run_batch.py

# 3. Tables, statistics and figures.
uv run --group analysis python examples/d_design_priors/analyse_design_priors.py
```

To watch the best robot of a run, use MuJoCo's viewer (Backspace restarts the simulation):

```bash
uv run python examples/d_design_priors/view_best.py ~/ariel_runs/design_priors/symmetric/seed_01.db
```

To save a video with a camera that follows the robot, pass a folder with `--video`. The camera films from behind by default; add `--view side` to film from the side. The video uses ARIEL's `tracking_video_renderer`.

```bash
uv run python examples/d_design_priors/view_best.py ~/ariel_runs/design_priors/symmetric/seed_01.db --video ~/ariel_runs/videos
```

The script shows the body with the highest score unless `--uid` picks another one. It replays the stored brain exactly as it was scored, and prints the recomputed score as a check. The video covers the scored 3 s by default; brains were never optimized beyond that. `--seconds` lengthens it.

To run a single run or a quick smoke test:

```bash
uv run python examples/d_design_priors/run_design_priors.py \
    --condition symmetric --seed 1 --pop 4 --gens 2 --learn-budget 1 --duration 0.5
```

## Outputs

Results go to `~/ariel_runs/design_priors/`. Keep them out of OneDrive, where syncing can corrupt SQLite files.

- `<condition>/seed_NN.db`: the EA database. Tags per body include its origin, seed ancestry, module and hinge counts, symmetry, score, learning curve, simulations used and the learned brain.
- `<condition>/seed_NN_meta.json`: settings, versions, status and wall time.
- `<condition>/seed_NN_reeval.json`: the best 3 bodies re-learned with 3 fresh seeds each.
- `analysis_config.json`: the pilot's measurements, recommended `--gens` and `tau_fit`.
- `analysis/`:
  - `runs.csv`, `stats.csv`, `summary.md`;
  - `best_so_far.png`, `primary_metrics.png`, `morphology.png`.

## Notes

- **Primary metrics:**
  - the mean best-so-far score over the simulation budget (area under the curve);
  - the re-evaluated final best.
- **Statistics:** Kruskal–Wallis across all conditions. Each prior is also tested against the baseline with a two-sided Mann–Whitney U test (Holm-corrected) and Vargha–Delaney A12. With 10 seeds per condition, only large effects (A12 ≳ 0.8) are detectable.
- **Speed:** on an M2 (4 performance + 4 efficiency cores) the pilot measured about 175 simulations/s with 7 parallel runs. Each run needs about 420 MB of memory; use `--jobs 6` if the machine starts swapping.
- **Reproducibility:** the same seed gives identical runs, whether bodies are evaluated in-process or by `--eval-workers`.
- **Interpreting `seeded`:** 4 of the 8 seeds are symmetric and 5 have at most 8 hinges, so read `seeded` as a bundle of design knowledge.
