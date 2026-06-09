"""Клієнт передавання Wi-Fi налаштувань для setup-точки ESP32-CAM."""

from __future__ import annotations

from urllib.parse import urlsplit

import requests

from config.settings import (
    DEFAULT_SETUP_HOST,
    DEFAULT_WIFI_PATH,
    DEFAULT_WIFI_RESET_PATH,
    WIFI_TIMEOUT_SECONDS,
)


class WifiProvisioningError(RuntimeError):
    """Помилка передавання Wi-Fi налаштувань до ESP32-CAM."""


class WifiProvisioningClient:
    """Надсилати Wi-Fi облікові дані до setup-точки ESP32-CAM."""

    def __init__(
        self,
        setup_host: str = DEFAULT_SETUP_HOST,
        port: int = 80,
        timeout: float = WIFI_TIMEOUT_SECONDS,
    ) -> None:
        """Підготувати HTTP-сесію для точки передавання Wi-Fi налаштувань."""

        # setup_host може вводитися як IP або URL; зберігаємо нормалізовану адресу.
        self._setup_host = self._normalize_host(setup_host)
        self._port = port
        self._timeout = timeout

        # Одна session повторно використовує HTTP-з’єднання для /wifi і /wifi/reset.
        self._session = requests.Session()

    def send_credentials(self, ssid: str, password: str) -> None:
        """Надіслати SSID і пароль Wi-Fi до ESP32-CAM."""

        # SSID без пробілів на краях потрібний прошивці для Wi-Fi.begin().
        clean_ssid = ssid.strip()
        if not clean_ssid:
            raise ValueError("SSID cannot be empty.")
        try:
            # Пароль не trim-иться: пробіли можуть бути частиною реального пароля мережі.
            response = self._session.post(
                self._build_url(DEFAULT_WIFI_PATH),
                data={"ssid": clean_ssid, "password": password},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WifiProvisioningError(f"Wi-Fi provisioning failed: {exc}") from exc

    def reset_credentials(self) -> None:
        """Скинути збережені Wi-Fi налаштування ESP32-CAM."""

        try:
            # POST /wifi/reset не потребує confirm=1, бо підтвердження вже робить GUI.
            response = self._session.post(
                self._build_url(DEFAULT_WIFI_RESET_PATH),
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WifiProvisioningError(f"Wi-Fi reset failed: {exc}") from exc

    def _build_url(self, path: str) -> str:
        """Побудувати HTTP URL для setup-точки."""

        clean_path = path if path.startswith("/") else f"/{path}"
        return f"http://{self._setup_host}:{self._port}{clean_path}"

    @staticmethod
    def _normalize_host(host: str) -> str:
        """Очистити адресу setup-вузла від схеми та шляху."""

        # Користувач може вставити http://192.168.4.1/wifi; клієнту потрібен тільки host.
        clean_host = host.strip()
        if not clean_host:
            raise ValueError("Setup host cannot be empty.")
        if "://" in clean_host:
            parsed = urlsplit(clean_host)
            clean_host = parsed.netloc or parsed.path
        clean_host = clean_host.split("/", 1)[0].strip()
        if not clean_host:
            raise ValueError("Setup host cannot be empty.")
        return clean_host
