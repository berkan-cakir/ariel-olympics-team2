# Gecko controller

Adapted from [baby-robot-homing](https://github.com/A-lamo/baby-robot-homing) (`hardware/baby_hardware.py`, `hardware/run_nn_homing.py`); follow its [hardware notes](https://github.com/A-lamo/baby-robot-homing/blob/main/wiki/hardware_implementation_notes.md) for the Pi setup and assembly.

| file | content |
|---|---|
| `gecko_103037.npz` | weights (4 KB): 11 inputs → 16 ELU → 8 tanh × π/2, then `command = 0.5 · previous + 0.5 · output` as ARIEL's `Controller` does |
| `gecko_hardware.py` | servo mapping, IMU axes, Robohat wrapper |
| `run_gecko.py` | controller, 10 Hz loop, CSV log, sweep and IMU check |

Inputs: the core's tilt from the accelerometer (quaternion x, y, z) and the 8 measured joint angles, both in this joint order: neck, spine, fl_leg, fl_flipper, fr_leg, fr_flipper, bl_leg, br_leg.

## On the Pi

```bash
git clone --filter=blob:none --sparse --branch Iris git@github.com:berkan-cakir/ariel-olympics-team2.git
cd ariel-olympics-team2
git sparse-checkout set research/hardware
```

The Robohat library is expected in `~/robohat` (with `robohatlib/` and `testlib/`), as in the hardware notes.

## Calibration

Edit `DEFAULT_SERVO_MAPPINGS` in `gecko_hardware.py`; the channels there are placeholders.

1. **Channels:** `channel` is the PWM port of each joint. Set `readback_channel` if its feedback is read on another channel.
2. **Neutral:** `neutral_deg` per servo (notes, section 2 step 5).
3. **Signs:** `python3 research/hardware/run_gecko.py --sweep 30` moves each joint to −30°, +30° and 0. At +30° the joint must move as in the table; if it moves the other way, set `sign=-1.0`. The printed readback must match the joint angle within a few degrees, otherwise fix `readback_channel`.

   | joint | positive angle in the simulation |
   |---|---|
   | neck | everything behind the neck swings toward the fl side |
   | spine | the rear part swings toward the fl side |
   | fl_leg | the arm lifts |
   | fl_flipper | the flipper swings toward the tail |
   | fr_leg | the arm goes down |
   | fr_flipper | the flipper swings toward the tail |
   | bl_leg | the leg moves up and toward the head |
   | br_leg | the leg moves down and toward the tail |

   The head is the core (Robohat box), the tail is the end with the back legs, the fl side is the side of `fl_leg`.

4. **IMU:** `python3 research/hardware/run_gecko.py --check-imu --duration 30` (servos stay off).
   - Robot flat: `acc` ≈ (0, 0, +9.8), `quat_imag` ≈ (0, 0, 0).
   - Lift the tail: the first `quat_imag` number goes positive.
   - Lower the fr side: the second number goes positive.

   If not, change `IMU_AXES` (which IMU axis is x, y, z) and `IMU_SIGNS`.

## Run

```bash
python3 research/hardware/run_gecko.py
```

1 s in neutral, 30 s of walking, back to neutral; Ctrl-C stops safely. Every step is logged to `~/gecko_logs/`. "late by" means the loop can't keep 10 Hz.
It walks head first. `min_deg`/`max_deg` (10–170) limit the controller's ±90° to ±80°.
Keep the servo readback on: in the simulation, `--no-readback` walks 0–1.5 m in 30 s.

# Spider controller

| file | content |
|---|---|
| `spider_54774.npz` | weights (10 KB): 11 inputs → 32 tanh → 32 tanh → 8 tanh × π/2, with the same 0.5 smoothing |
| `spider_hardware.py` | servo mapping, IMU axes, Robohat wrapper |
| `run_spider.py` | the same loop as `run_gecko.py`, for a network with any number of layers |

Joint order: l1_1, l1_2, l2_1, l2_2, l3_1, l3_2, l4_1, l4_2 (`_1` is the hip, `_2` the knee of each leg). Pi setup as for the gecko.

## Calibration

The same steps, with `DEFAULT_SERVO_MAPPINGS` in `spider_hardware.py` and `python3 research/hardware/run_spider.py --sweep 30` / `--check-imu --duration 30`.

- Seen from above, the legs go counterclockwise l1, l2, l4, l3 (l1 opposite l4, l2 opposite l3).
- Signs: at a positive angle every hip (`_1`) swings its leg counterclockwise seen from above, and every knee (`_2`) lifts its leg.
- IMU: robot flat gives `quat_imag` ≈ (0, 0, 0). Lifting the l1 side makes the first number positive; lowering the l3 side makes the second number positive.

## Run

```bash
python3 research/hardware/run_spider.py
```

Logs go to `~/spider_logs/`. It walks diagonally, toward the gap between legs l1 and l2. Keep the servo readback on: in the simulation, `--no-readback` walks 0.6–2.1 m in 30 s.
