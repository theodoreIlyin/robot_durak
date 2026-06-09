"""UDP-пошук ESP32-CAM роботів у локальній мережі."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from config.settings import DISCOVERY_PORT, DISCOVERY_SIGNATURE, DISCOVERY_TIMEOUT_SECONDS


class RobotDiscoveryError(RuntimeError):
    """Помилка некоректного discovery-повідомлення."""


@dataclass(frozen=True)
class RobotDiscoveryResult:
    """Мережеві параметри, які оголошує ESP32-CAM робот."""

    robot_id: str
    ip: str
    websocket_port: int
    stream_port: int
    name: str = "KPI Robot Vision Car"

    @property
    def display_name(self) -> str:
        """Повернути зрозумілу назву робота для списку в інтерфейсі."""

        return f"{self.name} — {self.ip} — ID {self.robot_id}"


def discover_robots(timeout: float = DISCOVERY_TIMEOUT_SECONDS) -> list[RobotDiscoveryResult]:
    """Зібрати всі UDP broadcast-повідомлення роботів за вказаний час."""

    # UDP socket слухає локальні broadcast-пакети, які ESP32 надсилає раз на інтервал.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Словник одночасно зберігає результат і прибирає дублікати за robot_id.
    robots: dict[str, RobotDiscoveryResult] = {}
    deadline = time.monotonic() + timeout

    try:
        # Порожня адреса означає прийом пакетів на всіх мережевих інтерфейсах ПК.
        sock.bind(("", DISCOVERY_PORT))

        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                continue

            message = data.decode("utf-8", errors="replace").strip()

            # Ігноруємо сторонні UDP-пакети, щоб не парсити нерелевантний трафік.
            if not message.startswith(DISCOVERY_SIGNATURE):
                continue

            try:
                result = _parse_discovery_message(message)
            except RobotDiscoveryError:
                continue

            # Якщо id відсутній у старому повідомленні, IP лишається резервним ключем.
            key = result.robot_id or result.ip
            robots[key] = result

    finally:
        sock.close()

    return sorted(robots.values(), key=lambda item: (item.name, item.ip))


def discover_robot(timeout: float = DISCOVERY_TIMEOUT_SECONDS) -> RobotDiscoveryResult | None:
    """Повернути першого знайденого робота для зворотної сумісності."""

    robots = discover_robots(timeout)
    if not robots:
        return None

    return robots[0]


def _parse_discovery_message(message: str) -> RobotDiscoveryResult:
    """Розібрати discovery-повідомлення ESP32-CAM у структурований результат."""

    # Формат повідомлення: KPI_ROBOT_CAR;key=value;key=value...
    parts = message.split(";")
    values: dict[str, str] = {}

    for part in parts[1:]:
        if "=" not in part:
            continue

        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()

    # IP є мінімально потрібним полем для підключення GUI до робота.
    ip = values.get("ip")
    if not ip:
        raise RobotDiscoveryError(f"Discovery message does not contain IP: {message}")

    # robot_id стабілізує вибір між кількома роботами навіть при повторних пакетах.
    robot_id = values.get("id") or ip

    try:
        # Порти мають тип int, бо їх напряму підставляє GUI у QSpinBox.
        websocket_port = int(values.get("ws", "80"))
        stream_port = int(values.get("stream", "81"))
    except ValueError as exc:
        raise RobotDiscoveryError(f"Invalid port in discovery message: {message}") from exc

    name = values.get("name", f"Robot-{robot_id[-4:]}")

    return RobotDiscoveryResult(
        robot_id=robot_id,
        ip=ip,
        websocket_port=websocket_port,
        stream_port=stream_port,
        name=name,
    )
