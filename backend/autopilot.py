"""Автопілот: пошук цільового об'єкта за кольором/формою/YOLO та генерація команд руху."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QMutex, QMutexLocker, QTimer, pyqtSignal, pyqtSlot

try:
    from ultralytics import YOLO  # type: ignore[import]
except Exception:  # noqa: BLE001
    YOLO = None

__all__ = [
    "AutoPilotConfig",
    "ColorFollowAutoPilot",
    "PinMissionState",
]


class PinMissionState(str, Enum):
    SEEK = "seek"
    RAM = "ram"
    BACKUP = "backup"
    SCAN_LEFT = "scan_left"
    SCAN_PAUSE = "scan_pause"
    SCAN_RIGHT = "scan_right"
    MISSION_DONE = "mission_done"


@dataclass
class AutoPilotConfig:
    # HSV-діапазон цільового кольору
    lower_hsv: tuple[int, int, int]
    upper_hsv: tuple[int, int, int]
    secondary_lower_hsv: Optional[tuple[int, int, int]] = None
    secondary_upper_hsv: Optional[tuple[int, int, int]] = None

    roi_top_ratio: float = 0.25
    min_area: int = 150
    mask_preview_enabled: bool = True
    frame_skip: int = 2
    lost_frames_threshold: int = 4

    shape_filter_enabled: bool = True
    min_circularity: float = 0.08
    min_rectangularity: float = 0.40
    max_aspect_ratio: float = 4.0
    max_candidates: int = 8

    detector_enabled: bool = False
    detector_model: str = "yolov8n.pt"
    detector_img_size: int = 416
    # list — стандартний патерн для mutable default у dataclass.
    detector_target_classes: Sequence[str] = field(default_factory=list)

    center_dead_zone: float = 0.18
    too_close_area_ratio: float = 0.28

    # Режим збивання кегель
    pin_mode_enabled: bool = False
    standing_height_ratio: float = 1.1   # h/w: стояча кегля вища за широку
    hit_area_ratio: float = 0.10         # почати таран, коли кегля велика в кадрі
    knocked_zone_radius: float = 0.18    # нормалізований радіус «вже збито»
    knock_lost_frames: int = 3           # кадрів без стоячої цілі після зближення
    action_seconds: float = 5.0        # кожна дія (назад, огляд, пауза) = 5 с
    max_ram_seconds: float = 1.4


class ColorFollowAutoPilot(QObject):
    """Обробляє кадри, шукає ціль і віддає команду руху."""

    command_ready = pyqtSignal(str)
    target_detected = pyqtSignal(object)
    mask_ready = pyqtSignal(object)
    mission_status = pyqtSignal(str)
    mission_complete = pyqtSignal()

    def __init__(self, config: AutoPilotConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._cfg = config
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_mutex = QMutex()
        self._frame_counter = 0
        self._last_command: Optional[str] = None
        self._lost_frames = 0

        self._pin_state = PinMissionState.SEEK
        self._state_until = 0.0
        self._knocked_zones: list[tuple[float, float]] = []
        self._pins_knocked = 0
        self._ram_started_at = 0.0
        self._close_tracking = False
        self._close_lost_frames = 0
        self._last_close_target: Optional[dict] = None
        self._last_mission_status = ""
        self._scan_pause_after = "right"

        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._process_latest_frame)

        self._detector = None
        self._detector_available = False
        if self._cfg.detector_enabled and YOLO is not None:
            try:
                self._detector = YOLO(self._cfg.detector_model)
                self._detector_available = True
            except Exception:  # noqa: BLE001
                self._detector = None
                self._detector_available = False

    def start(self) -> None:
        self._reset_pin_mission()
        self._timer.start()
        self._emit_mission_status("Автопілот запущено")

    def stop(self) -> None:
        self._timer.stop()
        self._last_command = None
        self._lost_frames = 0
        self._reset_pin_mission()
        self.target_detected.emit(None)
        self.command_ready.emit("stop")

    def update_config(self, config: AutoPilotConfig) -> None:
        pin_mode_changed = self._cfg.pin_mode_enabled != config.pin_mode_enabled
        self._cfg = config
        if pin_mode_changed:
            self._reset_pin_mission()
            self._last_command = None
        if self._cfg.detector_enabled and YOLO is not None and not self._detector_available:
            try:
                self._detector = YOLO(self._cfg.detector_model)
                self._detector_available = True
            except Exception:  # noqa: BLE001
                self._detector = None
                self._detector_available = False
        if not self._cfg.detector_enabled:
            self._detector = None
            self._detector_available = False

    @pyqtSlot(object)
    def on_frame(self, frame: np.ndarray) -> None:
        # Захист від ZeroDivisionError при frame_skip <= 0.
        if self._cfg.frame_skip <= 0:
            return
        self._frame_counter += 1
        if self._frame_counter % self._cfg.frame_skip != 0:
            return
        with QMutexLocker(self._frame_mutex):
            self._latest_frame = frame.copy()

    def _reset_pin_mission(self) -> None:
        self._pin_state = PinMissionState.SEEK
        self._state_until = 0.0
        self._knocked_zones = []
        self._pins_knocked = 0
        self._ram_started_at = 0.0
        self._close_tracking = False
        self._close_lost_frames = 0
        self._last_close_target = None
        self._last_mission_status = ""
        self._scan_pause_after = "right"

    def _emit_mission_status(self, text: str) -> None:
        if text == self._last_mission_status:
            return
        self._last_mission_status = text
        self.mission_status.emit(text)

    def _process_latest_frame(self) -> None:
        with QMutexLocker(self._frame_mutex):
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()

        command, target, mask = self._decide_command(frame)
        self.target_detected.emit(target)
        if self._cfg.mask_preview_enabled and mask is not None:
            self.mask_ready.emit(mask)

        if command is None:
            return

        if command == "stop":
            self._lost_frames += 1
            if (
                self._lost_frames < self._cfg.lost_frames_threshold
                and self._last_command is not None
                and self._last_command != "stop"
            ):
                return
        else:
            self._lost_frames = 0

        if command == self._last_command:
            return

        self._last_command = command
        self.command_ready.emit(command)

    def _build_color_mask(self, hsv: np.ndarray) -> np.ndarray:
        lower = np.array(self._cfg.lower_hsv, dtype=np.uint8)
        upper = np.array(self._cfg.upper_hsv, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        if self._cfg.secondary_lower_hsv and self._cfg.secondary_upper_hsv:
            lower2 = np.array(self._cfg.secondary_lower_hsv, dtype=np.uint8)
            upper2 = np.array(self._cfg.secondary_upper_hsv, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _full_frame_mask(self, roi_mask: np.ndarray, roi_top: int, frame_shape: tuple[int, int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        full = np.zeros((h, w), dtype=np.uint8)
        full[roi_top:h, :] = roi_mask
        return full

    def _decide_command(
        self, frame: np.ndarray
    ) -> tuple[Optional[str], Optional[dict], Optional[np.ndarray]]:
        h, w = frame.shape[:2]
        roi_top = int(h * self._cfg.roi_top_ratio)
        roi = frame[roi_top:h, :, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_mask = self._build_color_mask(hsv)
        preview_mask = (
            self._full_frame_mask(roi_mask, roi_top, frame.shape)
            if self._cfg.mask_preview_enabled
            else None
        )

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = self._filter_candidates(contours, roi_top, frame.shape) if contours else []

        if self._cfg.pin_mode_enabled:
            return self._decide_pin_mission(frame, candidates, preview_mask)

        return self._decide_color_follow(frame, candidates, preview_mask)

    def _decide_color_follow(
        self,
        frame: np.ndarray,
        candidates: list[dict],
        preview_mask: Optional[np.ndarray],
    ) -> tuple[Optional[str], Optional[dict], Optional[np.ndarray]]:
        if not candidates:
            return "stop", None, preview_mask

        candidates.sort(key=lambda c: c["area"], reverse=True)
        target = dict(candidates[0])
        target["confirmed"] = True

        if self._cfg.detector_enabled and self._detector_available:
            if not self._validate_with_detector(frame, target):
                target["confirmed"] = False
                return "stop", target, preview_mask

        if target["area_ratio"] > self._cfg.too_close_area_ratio:
            return "stop", target, preview_mask

        return self._steer_toward(target, frame.shape[1]), target, preview_mask

    def _decide_pin_mission(
        self,
        frame: np.ndarray,
        candidates: list[dict],
        preview_mask: Optional[np.ndarray],
    ) -> tuple[Optional[str], Optional[dict], Optional[np.ndarray]]:
        now = time.monotonic()
        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        self._remember_fallen_pins(candidates, frame_w, frame_h)

        standing = self._standing_pins(candidates)
        standing = [c for c in standing if not self._in_knocked_zone(c, frame_w, frame_h)]
        standing.sort(key=lambda c: (c["area"], -abs(c["cx"] - frame_w / 2)), reverse=True)

        target = dict(standing[0]) if standing else None
        if target is not None:
            target["confirmed"] = True
            target["pin_state"] = "standing"

        if self._pin_state == PinMissionState.MISSION_DONE:
            return "stop", None, preview_mask

        if self._pin_state == PinMissionState.BACKUP:
            if now < self._state_until:
                self._emit_mission_status(
                    f"Від'їзд після збиття ({self._pins_knocked} кегль)"
                )
                return "backward", target, preview_mask
            self._begin_scan_left(now)
            return "left", target, preview_mask

        if self._pin_state == PinMissionState.SCAN_LEFT:
            if standing:
                self._on_pin_found_during_scan()
                return self._approach_found_pin(standing[0], frame_w, target, preview_mask)
            if now < self._state_until:
                return "left", None, preview_mask
            self._begin_scan_pause(now, after="right")
            return "stop", None, preview_mask

        if self._pin_state == PinMissionState.SCAN_PAUSE:
            if standing:
                self._on_pin_found_during_scan()
                return self._approach_found_pin(standing[0], frame_w, target, preview_mask)
            if now < self._state_until:
                return "stop", None, preview_mask
            if self._scan_pause_after == "right":
                self._begin_scan_right(now)
                return "right", None, preview_mask
            self._finish_mission()
            return "stop", None, preview_mask

        if self._pin_state == PinMissionState.SCAN_RIGHT:
            if standing:
                self._on_pin_found_during_scan()
                return self._approach_found_pin(standing[0], frame_w, target, preview_mask)
            if now < self._state_until:
                return "right", None, preview_mask
            self._begin_scan_pause(now, after="finish")
            return "stop", None, preview_mask

        if self._pin_state == PinMissionState.RAM:
            knocked = self._check_pin_knocked(standing)
            ram_timeout = (now - self._ram_started_at) >= self._cfg.max_ram_seconds
            if knocked or ram_timeout:
                self._register_knocked_pin(frame_w, frame_h)
                self._begin_backup(now)
                return "backward", self._last_close_target, preview_mask

            if standing:
                self._track_close_approach(standing[0])
                ram_target = dict(standing[0])
                ram_target["pin_state"] = "ram"
                self._emit_mission_status("Збивання кеглі!")
                return "forward", ram_target, preview_mask

            # _close_lost_frames вже збільшено у _check_pin_knocked() вище,
            # тому тут додатковий інкремент не потрібен (він призводив до подвоєння).
            if self._close_tracking and self._close_lost_frames >= self._cfg.knock_lost_frames:
                self._register_knocked_pin(frame_w, frame_h)
                self._begin_backup(now)
                return "backward", self._last_close_target, preview_mask

            self._emit_mission_status("Збивання кеглі!")
            return "forward", self._last_close_target, preview_mask

        # SEEK
        if not standing:
            self._close_tracking = False
            self._close_lost_frames = 0
            self._begin_scan_left(now)
            return "left", None, preview_mask

        seek_target = standing[0]
        self._track_close_approach(seek_target)

        if seek_target["area_ratio"] >= self._cfg.hit_area_ratio:
            self._pin_state = PinMissionState.RAM
            self._ram_started_at = now
            self._close_lost_frames = 0
            ram_target = dict(seek_target)
            ram_target["pin_state"] = "ram"
            self._emit_mission_status("Під'їзд до кеглі — збивання")
            return "forward", ram_target, preview_mask

        self._emit_mission_status(
            f"Їду до кеглі ({self._pins_knocked} вже збито)"
        )
        return self._steer_toward(seek_target, frame_w), target, preview_mask

    def _steer_toward(self, target: dict, frame_w: int) -> str:
        center_x = frame_w / 2.0
        dx = (target["cx"] - center_x) / center_x
        if dx > self._cfg.center_dead_zone:
            return "right"
        if dx < -self._cfg.center_dead_zone:
            return "left"
        return "forward"

    def _is_standing_pin(self, candidate: dict) -> bool:
        return candidate["h"] >= candidate["w"] * self._cfg.standing_height_ratio

    def _standing_pins(self, candidates: list[dict]) -> list[dict]:
        return [c for c in candidates if self._is_standing_pin(c)]

    def _normalized_center(self, candidate: dict, frame_w: int, frame_h: int) -> tuple[float, float]:
        return candidate["cx"] / frame_w, candidate["cy"] / frame_h

    def _in_knocked_zone(self, candidate: dict, frame_w: int, frame_h: int) -> bool:
        if not self._knocked_zones:
            return False
        nx, ny = self._normalized_center(candidate, frame_w, frame_h)
        radius = self._cfg.knocked_zone_radius
        radius_sq = radius * radius
        for kx, ky in self._knocked_zones:
            dx = nx - kx
            dy = ny - ky
            if dx * dx + dy * dy <= radius_sq:
                return True
        return False

    def _remember_fallen_pins(self, candidates: list[dict], frame_w: int, frame_h: int) -> None:
        """Лежачі плями кольору біля останньої цілі теж позначаємо як «вже збито»."""
        if self._last_close_target is None:
            return
        last_nx = self._last_close_target["cx"] / frame_w
        last_ny = self._last_close_target["cy"] / frame_h
        zone_sq = (self._cfg.knocked_zone_radius * 1.3) ** 2

        for candidate in candidates:
            if self._is_standing_pin(candidate):
                continue
            if self._in_knocked_zone(candidate, frame_w, frame_h):
                continue
            if candidate["area_ratio"] < 0.015:
                continue
            nx, ny = self._normalized_center(candidate, frame_w, frame_h)
            dx = nx - last_nx
            dy = ny - last_ny
            if dx * dx + dy * dy <= zone_sq:
                self._knocked_zones.append((nx, ny))

    def _track_close_approach(self, target: dict) -> None:
        self._last_close_target = dict(target)
        if target["area_ratio"] >= self._cfg.hit_area_ratio * 0.7:
            self._close_tracking = True
            self._close_lost_frames = 0

    def _check_pin_knocked(self, standing: list[dict]) -> bool:
        if standing:
            self._close_lost_frames = 0
            return False
        if not self._close_tracking:
            return False
        self._close_lost_frames += 1
        return self._close_lost_frames >= self._cfg.knock_lost_frames

    def _register_knocked_pin(self, frame_w: int, frame_h: int) -> None:
        if self._last_close_target is None:
            return
        nx = self._last_close_target["cx"] / frame_w
        ny = self._last_close_target["cy"] / frame_h
        self._knocked_zones.append((nx, ny))
        self._pins_knocked += 1
        self._close_tracking = False
        self._close_lost_frames = 0
        self._emit_mission_status(f"Кеглю збито! Всього: {self._pins_knocked}")

    def _action_duration(self) -> float:
        return self._cfg.action_seconds

    def _begin_backup(self, now: float) -> None:
        self._pin_state = PinMissionState.BACKUP
        self._state_until = now + self._action_duration()
        self._last_command = None
        self._emit_mission_status(
            f"Їду назад {self._action_duration():.0f} с після збиття"
        )

    def _begin_scan_left(self, now: float) -> None:
        self._pin_state = PinMissionState.SCAN_LEFT
        self._state_until = now + self._action_duration()
        self._last_command = None
        self._emit_mission_status(
            f"Огляд ліворуч {self._action_duration():.0f} с "
            f"(збито: {self._pins_knocked})"
        )

    def _begin_scan_pause(self, now: float, after: str) -> None:
        self._pin_state = PinMissionState.SCAN_PAUSE
        self._state_until = now + self._action_duration()
        self._scan_pause_after = after
        self._last_command = None
        if after == "right":
            self._emit_mission_status(
                f"Стоп {self._action_duration():.0f} с — потім огляд праворуч"
            )
        else:
            self._emit_mission_status(
                f"Стоп {self._action_duration():.0f} с — шукаю наступну кеглю"
            )

    def _begin_scan_right(self, now: float) -> None:
        self._pin_state = PinMissionState.SCAN_RIGHT
        self._state_until = now + self._action_duration()
        self._last_command = None
        self._emit_mission_status(
            f"Огляд праворуч {self._action_duration():.0f} с"
        )

    def _on_pin_found_during_scan(self) -> None:
        self._pin_state = PinMissionState.SEEK
        self._close_tracking = False
        self._last_command = None

    def _approach_found_pin(
        self,
        pin: dict,
        frame_w: int,
        target: Optional[dict],
        preview_mask: Optional[np.ndarray],
    ) -> tuple[Optional[str], Optional[dict], Optional[np.ndarray]]:
        self._emit_mission_status("Кеглю видно — їду до неї")
        return self._steer_toward(pin, frame_w), target, preview_mask

    def _finish_mission(self) -> None:
        self._pin_state = PinMissionState.MISSION_DONE
        self._emit_mission_status(
            f"Місію завершено! Збито кегель потрібного кольору: {self._pins_knocked}"
        )
        self.mission_complete.emit()

    def _filter_candidates(
        self,
        contours: Iterable[np.ndarray],
        roi_top: int,
        frame_shape: tuple[int, int, int],
    ) -> list[dict]:
        frame_h, frame_w, _ = frame_shape
        frame_area = float(frame_h * frame_w)

        candidates: list[dict] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self._cfg.min_area:
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)
            y_full = y + roi_top
            box_area = float(w_box * h_box)
            if box_area <= 0:
                continue

            # area_ratio базується на реальній площі контуру (а не bounding box),
            # щоб пороги too_close_area_ratio та hit_area_ratio були узгоджені з area.
            area_ratio = area / frame_area
            aspect = max(w_box / max(h_box, 1), h_box / max(w_box, 1))
            if aspect > self._cfg.max_aspect_ratio:
                continue

            if self._cfg.shape_filter_enabled:
                perim = float(cv2.arcLength(cnt, True))
                if perim <= 0:
                    continue
                circularity = 4.0 * np.pi * area / (perim * perim)
                rectangularity = area / box_area
                if (
                    circularity < self._cfg.min_circularity
                    and rectangularity < self._cfg.min_rectangularity
                ):
                    continue

            candidates.append(
                {
                    "x": x,
                    "y": y_full,
                    "w": w_box,
                    "h": h_box,
                    "cx": x + w_box / 2.0,
                    "cy": y_full + h_box / 2.0,
                    "area": area,
                    "area_ratio": area_ratio,
                }
            )

            if len(candidates) >= self._cfg.max_candidates:
                break

        return candidates

    def _validate_with_detector(self, frame: np.ndarray, target: dict) -> bool:
        if not self._detector_available or self._detector is None:
            return True

        try:
            results = self._detector.predict(
                frame,
                imgsz=self._cfg.detector_img_size,
                verbose=False,
            )
        except Exception:  # noqa: BLE001
            return True

        if not results:
            return False

        x_t, y_t, w_t, h_t = target["x"], target["y"], target["w"], target["h"]
        x2_t = x_t + w_t
        y2_t = y_t + h_t
        allowed_classes = set(cls.lower() for cls in self._cfg.detector_target_classes)

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            names = getattr(r, "names", {}) or {}
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):  # type: ignore[attr-defined]
                x1, y1, x2, y2 = [float(v) for v in box]
                label = str(names.get(int(cls), "")).lower()
                if allowed_classes and label not in allowed_classes:
                    continue
                cx_det = 0.5 * (x1 + x2)
                cy_det = 0.5 * (y1 + y2)
                if x_t <= cx_det <= x2_t and y_t <= cy_det <= y2_t:
                    return True
        return False
