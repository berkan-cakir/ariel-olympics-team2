import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from spider_hardware import SpiderHardware

HERE = Path(__file__).resolve().parent
ACTIVATIONS = {"tanh": np.tanh, "elu": lambda x: np.where(x > 0, x, np.expm1(x)), "relu": lambda x: np.maximum(x, 0)}


class SpiderBrain:
    def __init__(self, path: Path) -> None:
        ckpt = np.load(path)
        self.layers = [(ckpt[f"W{i}"], ckpt[f"b{i}"]) for i in range(1, sum(key.startswith("W") for key in ckpt.files) + 1)]
        self.activation = ACTIVATIONS[str(ckpt["activation"])]
        self.scale = float(ckpt["output_scale"])
        self.alpha = float(ckpt["alpha"])
        self.period = float(ckpt["control_period"])
        self.joints = [str(name) for name in ckpt["joints"]]
        self.command = np.zeros(len(self.joints))

    def __call__(self, quat_imag: np.ndarray, joints: np.ndarray) -> np.ndarray:
        x = np.concatenate([quat_imag, joints]).astype(np.float32)
        for w, b in self.layers[:-1]:
            x = self.activation(w @ x + b)
        w, b = self.layers[-1]
        target = np.tanh(w @ x + b) * self.scale
        self.command = np.clip((1 - self.alpha) * self.command + self.alpha * target, -np.pi / 2, np.pi / 2)
        return self.command


def tilt_quaternion(acc: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(acc)
    if not norm > 1e-6:
        return np.zeros(3)
    ax, ay, az = acc / norm
    roll, pitch = math.atan2(ay, az), math.atan2(-ax, math.hypot(ay, az))
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([sr * cp, cr * sp, -sr * sp])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=HERE / "spider_54774.npz")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--neutral-hold", type=float, default=1.0)
    parser.add_argument("--no-readback", action="store_true")
    parser.add_argument("--delay-mode", action="store_true")
    parser.add_argument("--check-imu", action="store_true")
    parser.add_argument("--sweep", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path, default=Path.home() / "spider_logs")
    return parser.parse_args()


def check_imu(robot: SpiderHardware, duration: float) -> None:
    start = time.monotonic()
    while time.monotonic() - start < duration:
        acc = robot.read_acceleration()
        print(f"acc {np.round(acc, 2)}  quat_imag {np.round(tilt_quaternion(acc), 3)}")
        time.sleep(0.2)


def sweep(robot: SpiderHardware, degrees: float, readback: bool) -> None:
    for index, mapping in enumerate(robot.mappings):
        for angle in (-degrees, degrees, 0.0):
            command = np.zeros(len(robot.mappings))
            command[index] = math.radians(angle)
            servo = robot.set_joints(command)[index]
            time.sleep(1.5)
            measured = f", readback {math.degrees(robot.read_joints()[index]):+6.1f} deg" if readback else ""
            print(f"{mapping.name:11s} joint {angle:+6.1f} deg -> servo {servo:6.1f} deg{measured}")


def run(robot: SpiderHardware, brain: SpiderBrain, args: argparse.Namespace) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    path = args.log_dir / f"spider_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "qx", "qy", "qz", *[f"{name}_in" for name in brain.joints], *[f"{name}_cmd" for name in brain.joints], *[f"{name}_servo_deg" for name in brain.joints]])
        start = time.monotonic()
        tick = start
        while tick - start < args.duration:
            quat = tilt_quaternion(robot.read_acceleration())
            joints = brain.command.copy() if args.no_readback else robot.read_joints()
            command = brain(quat, joints)
            servo = robot.set_joints(command)
            writer.writerow([f"{tick - start:.3f}", *np.round(quat, 5), *np.round(joints, 5), *np.round(command, 5), *np.round(servo, 2)])
            tick += brain.period
            late = time.monotonic() - tick
            if late > 0:
                print(f"late by {late:.3f} s at t={tick - start:.1f}")
                tick = time.monotonic()
            time.sleep(max(0.0, tick - time.monotonic()))
    print(f"log: {path}")


def main() -> None:
    args = parse_args()
    brain = SpiderBrain(args.weights)
    robot = SpiderHardware(servos=not args.check_imu, direct_mode=not args.delay_mode)
    print(f"battery {robot.battery()}%")
    try:
        if args.check_imu:
            check_imu(robot, args.duration)
            return
        robot.neutral()
        time.sleep(args.neutral_hold)
        if args.sweep:
            sweep(robot, args.sweep, not args.no_readback)
        else:
            run(robot, brain, args)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
