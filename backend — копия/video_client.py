"""Неблокуючий MJPEG-читач відео для PyQt-інтерфейсу."""

from __future__ import annotations

import time

import cv2
import numpy as np
import requests
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from config.settings import DEFAULT_STREAM_PATH, DEFAULT_STREAM_PORT, VIDEO_RECONNECT_DELAY_SECONDS

# Запобіжник від накопичення пошкодженого або надто великого MJPEG-кадру в пам’яті.
MAX_MJPEG_FRAME_BYTES = 2 * 1024 * 1024


def _extract_content_length(header: bytes) -> int | None:
    """Дістати Content-Length із заголовка MJPEG-частини."""

    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            try:
                return int(line.split(b":", 1)[1].strip())
            except ValueError:
                return None
    return None


class VideoStreamThread(QThread):
    """Читати MJPEG-кадри без OpenCV/FFmpeg для розбору HTTP-потоку."""

    # frame_ready переносить готовий кадр у GUI-потік.
    frame_ready = pyqtSignal(QImage)
    # cv_frame_ready відправляє сирий BGR-кадр (numpy.ndarray) для комп'ютерного зору.
    cv_frame_ready = pyqtSignal(object)
    # status_changed переносить текст стану.
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        host: str,
        stream_port: int = DEFAULT_STREAM_PORT,
        stream_path: str = DEFAULT_STREAM_PATH,
        parent: object | None = None,
    ) -> None:
        """Підготувати фоновий потік читання відео з ESP32-CAM."""

        super().__init__(parent)

        # Параметри потоку зберігаються окремо, щоб можна було перезапускати читання після помилки.
        self._host = host.strip()
        self._stream_port = stream_port
        self._stream_path = stream_path

        # _running керує циклом читання, а requestInterruption() дає Qt-сумісний сигнал зупинки.
        self._running = False

        # Session і Response закриваються окремо, бо stop() може викликатися з GUI-потоку.
        self._session: requests.Session | None = None
        self._response: requests.Response | None = None

    def stop(self) -> bool:
        """Попросити відеопотік завершитися і дочекатися зупинки."""

        self._running = False
        self.requestInterruption()
        self._close_stream()
        return self.wait(3000)

    def run(self) -> None:
        """Запустити цикл підключення, читання та перепідключення MJPEG."""

        self._running = True
        stream_url = self._build_stream_url()
        self.status_changed.emit(f"Підключення відеопотоку: {stream_url}")

        # Кожна ітерація читає один HTTP stream; після помилки цикл чекає і перепідключається.
        while self._running and not self.isInterruptionRequested():
            try:
                self._read_stream(stream_url)
            except requests.RequestException as exc:
                if self._running:
                    self.status_changed.emit(f"Помилка відеопотоку: {exc}")
            except Exception as exc:  # noqa: BLE001
                # Не даємо помилкам декодера зупинити цикл інтерфейсу.
                if self._running:
                    self.status_changed.emit(f"Помилка декодування відео: {exc}")
            finally:
                self._close_stream()

            if self._running and not self.isInterruptionRequested():
                time.sleep(VIDEO_RECONNECT_DELAY_SECONDS)

        self._close_stream()
        self.status_changed.emit("Відеопотік зупинено.")

    def _read_stream(self, stream_url: str) -> None:
        """Читати один HTTP MJPEG-потік і передавати готові QImage і сирі BGR-кадри."""

        self._session = requests.Session()
        self._response = self._session.get(
            stream_url,
            stream=True,
            timeout=(3.0, 10.0),
            headers={"Connection": "close"},
        )
        self._response.raise_for_status()
        self.status_changed.emit("Відеопотік підключено.")

        # Буфер накопичує байти між chunk-ами до повного multipart-заголовка і JPEG-тіла.
        buffer = bytearray()

        for chunk in self._response.iter_content(chunk_size=4096):
            if not self._running or self.isInterruptionRequested():
                break
            if not chunk:
                continue

            buffer.extend(chunk)

            while True:
                # MJPEG-частина відокремлює заголовки від JPEG порожнім рядком CRLF CRLF.
                header_end = buffer.find(b"\r\n\r\n")
                if header_end < 0:
                    if len(buffer) > 262144:
                        # Якщо заголовок не знайдено довго, лишаємо хвіст як шанс на межу кадру.
                        del buffer[:-4096]
                    break

                header = bytes(buffer[:header_end])
                content_length = _extract_content_length(header)
                if (
                    content_length is None
                    or content_length <= 0
                    or content_length > MAX_MJPEG_FRAME_BYTES
                ):
                    del buffer[: header_end + 4]
                    continue

                # Чекаємо, доки в буфері буде весь JPEG згідно з Content-Length.
                frame_start = header_end + 4
                frame_end = frame_start + content_length
                if len(buffer) < frame_end:
                    break

                jpg = bytes(buffer[frame_start:frame_end])
                del buffer[:frame_end]
                if not (jpg.startswith(b"\xff\xd8") and jpg.endswith(b"\xff\xd9")):
                    continue

                # OpenCV декодує JPEG у BGR, а QImage очікує RGB-байти.
                frame_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Спершу віддаємо сирий кадр для автопілота.
                self.cv_frame_ready.emit(frame.copy())

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb_frame.shape

                # copy() від’єднує QImage від пам’яті numpy-масиву перед передачею між потоками.
                image = QImage(
                    rgb_frame.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format_RGB888,
                )
                self.frame_ready.emit(image.copy())

    def _build_stream_url(self) -> str:
        """Побудувати HTTP URL для MJPEG-потоку."""

        path = self._stream_path if self._stream_path.startswith("/") else f"/{self._stream_path}"
        return f"http://{self._host}:{self._stream_port}{path}"

    def _close_stream(self) -> None:
        """Закрити HTTP-відповідь і сесію поточного відеопотоку."""

        if self._response is not None:
            self._response.close()
            self._response = None
        if self._session is not None:
            self._session.close()
            self._session = None