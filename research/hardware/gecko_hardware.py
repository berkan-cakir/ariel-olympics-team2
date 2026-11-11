import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROBOHAT = Path.home() / "robohat"


@dataclass(frozen=True)
class ServoMapping:
    name: str
    channel: int
    readback_channel: int | None = None
    neutral_deg: float = 90.0
    sign: float = 1.0
    min_deg: float = 10.0
    max_deg: float = 170.0


DEFAULT_SERVO_MAPPINGS = [
    ServoMapping("neck", 16),
    ServoMapping("spine", 17),
    ServoMapping("fl_leg", 0),
    ServoMapping("fl_flipper", 1),
    ServoMapping("fr_leg", 2),
    ServoMapping("fr_flipper", 3),
    ServoMapping("bl_leg", 18),
    ServoMapping("br_leg", 19),
]
IMU_AXES = (0, 1, 2)
IMU_SIGNS = (1.0, 1.0, 1.0)


class DisabledCamera:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def init_camera(self) -> None:
        pass

    def is_cam_available(self) -> bool:
        return False

    def get_capture_array(self) -> None:
        return None

    def test_camera(self) -> None:
        pass


def create_robohat():
    if str(ROBOHAT) not in sys.path:
        sys.path.insert(0, str(ROBOHAT))
    module = importlib.import_module("robohatlib.Robohat")
    module.Camera = DisabledCamera
    from testlib.TestConfig import SERVOASSEMBLY_1_CONFIG, SERVOASSEMBLY_2_CONFIG, SERVOBOARD_1_DATAS_LIST, SERVOBOARD_2_DATAS_LIST, TOPBOARD_ID_SWITCH

    robohat = module.Robohat(SERVOASSEMBLY_1_CONFIG, SERVOASSEMBLY_2_CONFIG, TOPBOARD_ID_SWITCH)
    robohat.init(SERVOBOARD_1_DATAS_LIST, SERVOBOARD_2_DATAS_LIST)
    return robohat


class GeckoHardware:
    def __init__(self, mappings: list[ServoMapping] = DEFAULT_SERVO_MAPPINGS, servos: bool = True, direct_mode: bool = True, delay_s: float = 0.02) -> None:
        self.mappings = mappings
        self.servos = servos
        self.robohat = create_robohat()
        self.robohat.do_buzzer_beep()
        self.degrees = [90.0] * 32
        if servos:
            self.robohat.start_servo_drivers()
            self.robohat.wakeup_servo()
            self.robohat.set_servo_direct_mode(direct_mode, delay_s)

    def set_joints(self, radians: np.ndarray) -> list[float]:
        for mapping, angle in zip(self.mappings, radians, strict=True):
            self.degrees[mapping.channel] = min(mapping.max_deg, max(mapping.min_deg, mapping.neutral_deg + mapping.sign * math.degrees(angle)))
        self.robohat.set_servo_multiple_angles(self.degrees)
        return [self.degrees[mapping.channel] for mapping in self.mappings]

    def read_joints(self) -> np.ndarray:
        degrees = [float(self.robohat.get_servo_single_angle(mapping.channel if mapping.readback_channel is None else mapping.readback_channel)) for mapping in self.mappings]
        return np.array([mapping.sign * math.radians(deg - mapping.neutral_deg) for mapping, deg in zip(self.mappings, degrees, strict=True)])

    def read_acceleration(self) -> np.ndarray:
        acc = self.robohat.get_imu_acceleration()
        return np.array([sign * float(acc[axis]) for axis, sign in zip(IMU_AXES, IMU_SIGNS, strict=True)])

    def battery(self) -> int:
        return int(self.robohat.get_battery_percentage_capacity())

    def neutral(self) -> None:
        self.set_joints(np.zeros(len(self.mappings)))

    def close(self) -> None:
        if self.servos:
            self.neutral()
            time.sleep(0.5)
            self.robohat.stop_servo_drivers()
        self.robohat.exit_program()
