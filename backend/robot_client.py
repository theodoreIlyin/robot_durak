"""WebSocket-клієнт для керування ESP32-CAM роботом."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

import requests
from websocket import (
    WebSocket,
    WebSocketConnectionClosedException,
    WebSocketException,
    WebSocketTimeoutException,
    create_connection,
)

from config.settings import (
    DEFAULT_ACTION_PATH,
    DEFAULT_SPEED,
    DEFAULT_WS_PATH,
    DEFAULT_WS_PORT,
    MAX_SPEED,
    MIN_SPEED,
    WS_CONNECT_TIMEOUT_SECONDS,
    WS_RESPONSE_TIMEOUT_SECONDS,
)

# Команди руху обмежені поточним WebSocket API прошивки.
VALID_MOTION_COMMANDS: Final[set[str]] = {"forward", "backward", "left", "right", "stop"}

# Команди руху не повторюються автоматично, щоб не продовжити рух після збою зв’язку.
CRITICAL_MOTION_COMMANDS: Final[set[str]] = {"forward", "backward", "left", "right", "stop"}

# Ці помилки означають, що socket можна закрити і для некритичної команди спробувати ще раз.
RECOVERABLE_WS_ERRORS: Final[tuple[type[BaseException], ...]] = (
    WebSocketConnectionClosedException,
    WebSocketTimeoutException,
    WebSocketException,
    OSError,
    TimeoutError,
)


class RobotClientError(RuntimeError):
    """Помилка виконання команди через WebSocket API робота."""


@dataclass(frozen=True)
class RobotCommandResult:
    """Короткий результат команди для статусних повідомлень інтерфейсу."""

    command: str
    status_code: int
    response_text: str


class RobotClient:
    """Надсилати команди руху та LED до ESP32-CAM через WebSocket API."""

    def __init__(
        self,
        host: str,
        command_port: int = DEFAULT_WS_PORT,
        timeout: float | tuple[float, float] = WS_RESPONSE_TIMEOUT_SECONDS,
        ws_path: str = DEFAULT_WS_PATH,
    ) -> None:
        """Підготувати клієнт для підключення до вказаного ESP32-CAM."""

        # _host зберігає лише адресу без схеми, шляху та порту для повторного складання URL.
        self._host = self._normalize_host(host)
        self._command_port = command_port
        self._connect_timeout, self._response_timeout = self._normalize_timeout(timeout)
        self._ws_path = ws_path

        # _speed є локальною копією останнього значення з GUI.
        # Прошивка отримує його тільки через send_speed().
        self._speed = DEFAULT_SPEED

        # WebSocket тримається відкритим між командами, щоб не створювати TCP-з’єднання щоразу.
        self._socket: WebSocket | None = None

    def set_speed(self, speed: int) -> None:
        """Зберегти швидкість у допустимих межах без негайного надсилання."""

        self._speed = self.clamp_speed(speed)

    def stop(self) -> RobotCommandResult:
        """Надіслати stop з резервним HTTP-запитом у разі помилки WebSocket."""

        try:
            return self.send_motion_command("stop")
        except RobotClientError:
            # Legacy HTTP /action лишається тільки як аварійний канал зупинки.
            self._send_stop_fallback_http()
            raise

    def send_motion_command(self, command: str, speed: int | None = None) -> RobotCommandResult:
        """Надіслати одну команду руху через WebSocket."""

        # Валідація не дає GUI або зовнішньому коду відправити неіснуючу команду руху.
        if command not in VALID_MOTION_COMMANDS:
            raise ValueError(f"Unsupported motion command: {command!r}")
        if speed is not None:
            self.set_speed(speed)
        return self._send_command(command, command)

    def send_speed(self, speed: int | None = None) -> RobotCommandResult:
        """Надіслати поточну або передану швидкість командою speed."""

        if speed is not None:
            self.set_speed(speed)
        return self._send_command(
            f"speed:{self._speed}",
            "speed",
            expected_response="OK",
        )

    def led_on(self) -> RobotCommandResult:
        """Увімкнути світлодіод ESP32-CAM."""

        return self._send_command("led:on", "led:on")

    def led_off(self) -> RobotCommandResult:
        """Вимкнути світлодіод ESP32-CAM."""

        return self._send_command("led:off", "led:off")

    def ping(self) -> RobotCommandResult:
        """Перевірити доступність WebSocket API командою ping."""

        return self._send_command("ping", "ping", expected_response="PONG")

    def status(self) -> RobotCommandResult:
        """Запитати поточний статус робота через WebSocket API."""

        return self._send_command("status", "status", expected_response=None)

    def close(self, send_stop: bool = True) -> None:
        """Закрити WebSocket-з’єднання, за потреби попередньо надіславши stop."""

        try:
            if send_stop and self._socket is not None:
                self.stop()
        except RobotClientError:
            pass
        finally:
            self._close_socket()

    def _send_command(
        self,
        payload: str,
        command: str,
        expected_response: str | None = "OK",
    ) -> RobotCommandResult:
        """Надіслати команду з коротким повтором для некритичних запитів."""

        attempts = 1 if command in CRITICAL_MOTION_COMMANDS else 2
        last_error: BaseException | None = None
        for _ in range(attempts):
            try:
                return self._send_once(payload, command, expected_response)
            except RobotClientError:
                # ERR:* від прошивки є валідною відповіддю API, тому повтор не допоможе.
                raise
            except RECOVERABLE_WS_ERRORS as exc:
                last_error = exc
                # Закриваємо socket перед повтором, щоб наступна спроба створила чисте з’єднання.
                self._close_socket()
        raise RobotClientError(
            f"WebSocket command '{command}' failed: {last_error}"
        ) from last_error

    def _send_once(
        self,
        payload: str,
        command: str,
        expected_response: str | None,
    ) -> RobotCommandResult:
        """Виконати одну WebSocket-відправку та перевірити відповідь."""

        socket = self._ensure_connection()
        socket.send(payload)
        response = socket.recv()
        response_text = str(response).strip()

        # Прошивка повертає ERR:* для синтаксично неправильних або недопустимих команд.
        if response_text.startswith("ERR:"):
            raise RobotClientError(f"WebSocket command '{command}' failed: {response_text}")
        if expected_response is not None and response_text != expected_response:
            # Для ping/speed/руху очікується конкретна коротка відповідь, інакше стан невідомий.
            raise RobotClientError(
                f"WebSocket command '{command}' failed: unexpected response {response_text!r}"
            )
        return RobotCommandResult(command, 200, response_text)

    def _ensure_connection(self) -> WebSocket:
        """Повернути активний WebSocket або створити нове підключення."""

        if self._socket is not None and self._socket.connected:
            return self._socket

        # Старий socket міг залишитись напіввідкритим після таймауту, тому закриваємо його явно.
        self._close_socket()
        socket = create_connection(self._build_ws_url(), timeout=self._connect_timeout)
        socket.settimeout(self._response_timeout)
        self._socket = socket
        return socket

    def _close_socket(self) -> None:
        """Закрити поточний WebSocket і очистити посилання на нього."""

        if self._socket is None:
            return
        try:
            self._socket.close()
        except Exception:
            pass
        finally:
            self._socket = None

    def _send_stop_fallback_http(self) -> None:
        """Спробувати зупинити робота через legacy HTTP-точку."""

        try:
            requests.get(
                f"http://{self._host}:{DEFAULT_WS_PORT}{DEFAULT_ACTION_PATH}",
                params={"go": "stop"},
                timeout=(0.5, 0.5),
            )
        except requests.RequestException:
            pass

    @staticmethod
    def clamp_speed(speed: int) -> int:
        """Обмежити швидкість допустимим PWM-діапазоном."""

        return max(MIN_SPEED, min(MAX_SPEED, int(speed)))

    def _build_ws_url(self) -> str:
        """Побудувати WebSocket URL для поточної адреси, порту та шляху."""

        clean_path = (
            self._ws_path
            if self._ws_path.startswith("/")
            else f"/{self._ws_path}"
        )
        return f"ws://{self._host}:{self._command_port}{clean_path}"

    @staticmethod
    def _normalize_timeout(timeout: float | tuple[float, float]) -> tuple[float, float]:
        """Нормалізувати timeout до пари connect/response."""

        if isinstance(timeout, tuple):
            return float(timeout[0]), float(timeout[1])
        return WS_CONNECT_TIMEOUT_SECONDS, float(timeout)

    @staticmethod
    def _normalize_host(host: str) -> str:
        """Очистити адресу вузла від схеми, шляху та порту для подальших URL."""

        # Користувач може вставити IP, http://IP, ws://IP/ws або IP:port; тут лишається тільки host.
        clean_host = host.strip()
        if not clean_host:
            raise ValueError("ESP32-CAM host cannot be empty.")
        if "://" in clean_host:
            parsed = urlsplit(clean_host)
            clean_host = parsed.hostname or parsed.netloc or parsed.path
        clean_host = clean_host.split("/", 1)[0].strip()
        if clean_host.startswith("[") and "]" in clean_host:
            clean_host = clean_host[1:].split("]", 1)[0]
        elif ":" in clean_host:
            clean_host = clean_host.split(":", 1)[0]
        if not clean_host:
            raise ValueError("ESP32-CAM host cannot be empty.")
        return clean_host
