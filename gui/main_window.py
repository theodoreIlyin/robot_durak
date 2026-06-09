"""Головне PyQt-вікно для керування ESP32-CAM роботом та автопілотом з комп'ютерним зором."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from PyQt5.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from backend.autopilot import AutoPilotConfig, ColorFollowAutoPilot
from backend.discovery import RobotDiscoveryResult, discover_robots
from backend.hsv_config import ColorRange, load_color_ranges, save_color_ranges
from backend.robot_client import RobotClient
from backend.video_client import VideoStreamThread
from backend.wifi import WifiProvisioningClient
from config.settings import (
    APP_NAME,
    DEFAULT_HOST,
    DEFAULT_SETUP_HOST,
    DEFAULT_SPEED,
    DEFAULT_STREAM_PORT,
    DEFAULT_WS_PORT,
    MAX_SPEED,
    MIN_SPEED,
    MOTION_KEEPALIVE_INTERVAL_MS,
    ROBOT_AUTO_RECONNECT_ENABLED,
    ROBOT_ERROR_THRESHOLD,
    ROBOT_RECONNECT_INTERVAL_MS,
    VIDEO_PLACEHOLDER_TEXT,
    WIFI_TIMEOUT_SECONDS,
)


class WorkerSignals(QObject):
    """Qt-сигнали для результатів фонової команди."""

    success = pyqtSignal(object)
    error = pyqtSignal(str)


class CommandWorker(QRunnable):
    """Виконувати коротку мережеву дію поза потоком інтерфейсу."""

    def __init__(self, action: Callable[[], Any]) -> None:
        """Зберегти дію, яку потрібно виконати у QThreadPool."""
        super().__init__()
        self.signals = WorkerSignals()
        self._action = action

    @pyqtSlot()
    def run(self) -> None:
        """Виконати дію та передати результат або помилку через Qt-сигнали."""
        try:
            result = self._action()
        except Exception as exc:  # noqa: BLE001 - інтерфейс показує мережеві помилки.
            self.signals.error.emit(str(exc))
        else:
            self.signals.success.emit(result)


class MainWindow(QMainWindow):
    """Головне вікно з підключенням, відео, ручним керуванням та автопілотом."""

    def __init__(self) -> None:
        """Ініціалізувати стан інтерфейсу, таймери та основні елементи."""
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(980, 700)

        # QThreadPool виконує короткі мережеві дії, щоб Qt event loop не блокувався.
        self._thread_pool = QThreadPool.globalInstance()

        # Базові посилання на активне підключення, відеопотік і знайдених роботів.
        self._connected = False
        self.robot_client: RobotClient | None = None
        self._video_thread: VideoStreamThread | None = None
        self._discovered_robots: list[RobotDiscoveryResult] = []

        # Стан руху і прапорці команд не дають одночасно відправляти несумісні WebSocket-запити.
        self._active_motion_command: str | None = None
        self._motion_command_in_flight = False
        self._stop_in_flight = False
        self._pending_stop = False
        self._led_in_flight = False
        self._speed_in_flight = False

        # Прапорці відключення і відновлення розрізняють ручне відключення та втрату зв’язку.
        self._disconnect_requested = False
        self._manual_disconnect_requested = False
        self._recovering_connection = False
        self._reconnect_attempts = 0
        self._reconnect_in_flight = False

        # Лічильник помилок запускає recovery лише після кількох послідовних збоїв.
        self._robot_error_count = 0
        self._max_robot_errors = ROBOT_ERROR_THRESHOLD

        # Keep-alive повторює активну команду руху частіше, ніж спрацьовує watchdog прошивки.
        self._motion_keepalive_timer = QTimer(self)
        self._motion_keepalive_timer.setInterval(MOTION_KEEPALIVE_INTERVAL_MS)
        self._motion_keepalive_timer.timeout.connect(self._repeat_motion_command)

        # Reconnect timer робить окремі спроби підключення після втрати зв’язку.
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setInterval(ROBOT_RECONNECT_INTERVAL_MS)
        self._reconnect_timer.timeout.connect(self._try_auto_reconnect)

        # Останній запис журналу зберігається, щоб не дублювати однакові статуси.
        self._last_log_entry: tuple[str, str] | None = None

        # ------------------------ Комп'ютерний зір / автопілот -------------------------
        self._color_ranges: dict[str, ColorRange] = load_color_ranges()
        # Поточний колір для автопілота (за замовчуванням оранжевий маркер, якщо є).
        self._current_color_name: str = (
            "orange" if "orange" in self._color_ranges else next(iter(self._color_ranges.keys()))
        )
        self._autopilot: ColorFollowAutoPilot | None = None
        self._autopilot_enabled: bool = False
        self._autopilot_target: dict | None = None
        self._autopilot_mask: object | None = None

        self._build_ui()
        self._connect_signals()
        self._set_connected(False)
        self._set_status("Готово. Вкажіть IP ESP32-CAM і натисніть «Підключитися».")

    # --------------------------------------------------------------------- Qt hooks

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - назва API Qt.
        """Коректно зупинити робота, відео та перепідключення перед закриттям."""
        self._manual_disconnect_requested = True
        self._reconnect_timer.stop()
        self._recovering_connection = False
        self._disconnect_requested = True
        self._active_motion_command = None
        self._motion_keepalive_timer.stop()
        if self._has_active_robot_request():
            self._pending_stop = True
        else:
            self._send_stop_blocking()
            self._close_robot_client(send_stop=False)
            self._stop_video()

        if self._video_thread is not None and self._video_thread.isRunning():
            event.ignore()
            self._set_status(
                "Відеопотік ще завершується. Повторіть закриття через кілька секунд.",
                level="WARNING",
            )
            return

        event.accept()

    # --------------------------------------------------------------------- UI build

    def _build_ui(self) -> None:
        """Побудувати основний макет головного вікна."""
        central_widget = QWidget(self)
        root_layout = QVBoxLayout(central_widget)

        # Верхній блок містить усі параметри мережевого підключення.
        root_layout.addWidget(self._build_connection_group())

        # Відео займає більшу частину ширини, а кнопки руху лишаються праворуч.
        content_layout = QHBoxLayout()
        content_layout.addWidget(self._build_video_group(), stretch=3)
        content_layout.addWidget(self._build_motion_group(), stretch=1)
        root_layout.addLayout(content_layout, stretch=1)
        root_layout.addWidget(self._build_log_group(), stretch=0)
        self.setCentralWidget(central_widget)

    def _build_connection_group(self) -> QGroupBox:
        """Створити блок параметрів підключення та Wi-Fi."""
        group = QGroupBox("Параметри підключення")
        layout = QGridLayout(group)

        # Поля SSID/пароля використовуються тільки для provisioning setup-точки ESP32-CAM.
        self.ssid_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)

        # setup_host може відрізнятися від host робота після discovery або ручного вводу.
        self.setup_host_edit = QLineEdit(DEFAULT_SETUP_HOST)
        self.host_edit = QLineEdit(DEFAULT_HOST)

        # Порти лишаються редагованими, бо discovery може повернути інші значення.
        self.command_port_spin = QSpinBox()
        self.command_port_spin.setRange(1, 65535)
        self.command_port_spin.setValue(DEFAULT_WS_PORT)
        self.stream_port_spin = QSpinBox()
        self.stream_port_spin.setRange(1, 65535)
        self.stream_port_spin.setValue(DEFAULT_STREAM_PORT)

        # ComboBox заповнюється UDP discovery і дозволяє вибрати одного з кількох роботів.
        self.robot_combo = QComboBox()
        self.robot_combo.setMinimumWidth(260)
        self.robot_combo.setToolTip("Список роботів, знайдених у поточній Wi-Fi мережі")

        # Швидкість надсилається окремою командою speed:, а не в кожній команді руху.
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(MIN_SPEED, MAX_SPEED)
        self.speed_spin.setValue(DEFAULT_SPEED)
        self.speed_spin.setSuffix(" PWM")

        form_left = QFormLayout()
        form_left.addRow("SSID:", self.ssid_edit)
        form_left.addRow("Пароль:", self.password_edit)
        form_left.addRow("Setup host:", self.setup_host_edit)

        form_right = QFormLayout()
        form_right.addRow("Знайдений робот:", self.robot_combo)
        form_right.addRow("IP ESP32-CAM:", self.host_edit)
        form_right.addRow("Порт команд:", self.command_port_spin)
        form_right.addRow("Порт відео:", self.stream_port_spin)
        form_right.addRow("Швидкість:", self.speed_spin)

        self.wifi_button = QPushButton("Передати Wi-Fi налаштування")
        self.wifi_reset_button = QPushButton("Скинути Wi-Fi")
        self.discovery_button = QPushButton("Знайти робота")
        self.connect_button = QPushButton("Підключитися")
        self.disconnect_button = QPushButton("Відключитися")

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.wifi_button)
        buttons_layout.addWidget(self.wifi_reset_button)
        buttons_layout.addWidget(self.discovery_button)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.connect_button)
        buttons_layout.addWidget(self.disconnect_button)

        layout.addLayout(form_left, 0, 0)
        layout.addLayout(form_right, 0, 1)
        layout.addLayout(buttons_layout, 1, 0, 1, 2)
        return group

    def _build_video_group(self) -> QGroupBox:
        """Створити блок перегляду MJPEG-відео."""
        group = QGroupBox("Відео з ESP32-CAM")
        layout = QVBoxLayout(group)

        self.video_label = QLabel(VIDEO_PLACEHOLDER_TEXT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setFrameShape(QFrame.StyledPanel)
        self.video_label.setStyleSheet("background-color: #111; color: #eee;")

        layout.addWidget(self.video_label)
        return group

    def _build_motion_group(self) -> QGroupBox:
        """Створити блок кнопок ручного керування + автопілот/HSV."""
        group = QGroupBox("Ручне керування / автопілот")
        layout = QGridLayout(group)

        # -------------------------- Ручні команди руху --------------------------
        self.forward_button = QPushButton("Вперед")
        self.backward_button = QPushButton("Назад")
        self.left_button = QPushButton("Ліворуч")
        self.right_button = QPushButton("Праворуч")
        self.stop_button = QPushButton("Стоп")
        self.led_button = QPushButton("Світлодіод: вимкнено")
        self.led_button.setCheckable(True)

        for button in (
            self.forward_button,
            self.backward_button,
            self.left_button,
            self.right_button,
            self.stop_button,
            self.led_button,
        ):
            button.setMinimumHeight(48)
        self.stop_button.setStyleSheet("font-weight: 600;")

        layout.addWidget(self.forward_button, 0, 1)
        layout.addWidget(self.left_button, 1, 0)
        layout.addWidget(self.stop_button, 1, 1)
        layout.addWidget(self.right_button, 1, 2)
        layout.addWidget(self.backward_button, 2, 1)
        layout.addWidget(self.led_button, 3, 0, 1, 3)

        # --------------------------- Блок автопілота / HSV ---------------------------
        self.autopilot_checkbox = QCheckBox("Автопілот: слідкувати за кольором")
        layout.addWidget(self.autopilot_checkbox, 4, 0, 1, 3)

        color_label = QLabel("Цільовий колір:")
        self.color_combo = QComboBox()
        for name in sorted(self._color_ranges.keys()):
            self.color_combo.addItem(name)
        idx = self.color_combo.findText(self._current_color_name)
        if idx >= 0:
            self.color_combo.setCurrentIndex(idx)

        layout.addWidget(color_label, 5, 0)
        layout.addWidget(self.color_combo, 5, 1, 1, 2)

        # HSV-спінбокси
        self.h_min_spin = QSpinBox()
        self.h_min_spin.setRange(0, 179)
        self.h_max_spin = QSpinBox()
        self.h_max_spin.setRange(0, 179)
        self.s_min_spin = QSpinBox()
        self.s_min_spin.setRange(0, 255)
        self.s_max_spin = QSpinBox()
        self.s_max_spin.setRange(0, 255)
        self.v_min_spin = QSpinBox()
        self.v_min_spin.setRange(0, 255)
        self.v_max_spin = QSpinBox()
        self.v_max_spin.setRange(0, 255)

        layout.addWidget(QLabel("H мін / макс:"), 6, 0)
        layout.addWidget(self.h_min_spin, 6, 1)
        layout.addWidget(self.h_max_spin, 6, 2)

        layout.addWidget(QLabel("S мін / макс:"), 7, 0)
        layout.addWidget(self.s_min_spin, 7, 1)
        layout.addWidget(self.s_max_spin, 7, 2)

        layout.addWidget(QLabel("V мін / макс:"), 8, 0)
        layout.addWidget(self.v_min_spin, 8, 1)
        layout.addWidget(self.v_max_spin, 8, 2)

        self.pin_mode_checkbox = QCheckBox("Режим кеглів (збивати по черзі)")
        self.pin_mode_checkbox.setChecked(False)
        self.pin_mode_checkbox.setToolTip(
            "Вимкніть для простого руху до вибраного кольору. Увімкнений режим "
            "після збиття від'їжджає назад, оглядається і шукає наступну кеглю."
        )
        layout.addWidget(self.pin_mode_checkbox, 9, 0, 1, 3)

        self.mask_preview_checkbox = QCheckBox("Показати маску кольору на відео")
        self.mask_preview_checkbox.setChecked(True)
        self.mask_preview_checkbox.setToolTip(
            "Накладає червоний шар там, де камера бачить обраний колір. "
            "Допомагає підібрати HSV-діапазон."
        )
        layout.addWidget(self.mask_preview_checkbox, 10, 0, 1, 3)

        # Додаткові опції фільтрації
        self.shape_filter_checkbox = QCheckBox("Фільтрація за формою")
        self.shape_filter_checkbox.setChecked(False)
        layout.addWidget(self.shape_filter_checkbox, 11, 0, 1, 3)

        self.detector_checkbox = QCheckBox("Детектор об'єктів (YOLO)")
        self.detector_checkbox.setToolTip(
            "Додаткова перевірка через YOLO. Для слідкування лише за кольором "
            "залиште вимкненим. Якщо увімкнено — ціль має збігатися і з кольором, "
            "і з класом YOLO (наприклад person)."
        )
        layout.addWidget(self.detector_checkbox, 12, 0, 1, 3)

        detector_classes_label = QLabel("Класи детектора (через кому):")
        self.detector_classes_edit = QLineEdit("")
        layout.addWidget(detector_classes_label, 13, 0, 1, 3)
        layout.addWidget(self.detector_classes_edit, 14, 0, 1, 3)

        # Кнопки застосування/збереження HSV
        self.apply_hsv_button = QPushButton("Застосувати до автопілота")
        self.save_hsv_button = QPushButton("Зберегти HSV в конфіг")
        layout.addWidget(self.apply_hsv_button, 15, 0, 1, 3)
        layout.addWidget(self.save_hsv_button, 16, 0, 1, 3)

        # Завантажуємо стартові значення для обраного кольору.
        self._load_hsv_to_ui(self._current_color_name)

        return group

    def _build_log_group(self) -> QGroupBox:
        """Створити блок журналу статусів і помилок."""
        group = QGroupBox("Журнал статусу і помилок")
        layout = QVBoxLayout(group)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QPlainTextEdit.NoWrap)

        # Ліміт рядків не дає журналу необмежено рости під час довгої сесії.
        self.log_edit.setMaximumBlockCount(1000)
        self.log_edit.setMinimumHeight(130)
        self.log_edit.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")

        self.copy_log_button = QPushButton("Копіювати журнал")
        self.clear_log_button = QPushButton("Очистити журнал")

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.copy_log_button)
        buttons_layout.addWidget(self.clear_log_button)

        layout.addWidget(self.log_edit)
        layout.addLayout(buttons_layout)
        return group

    # ----------------------------------------------------------------- сигналізація

    def _connect_signals(self) -> None:
        """Під’єднати Qt-сигнали елементів керування до обробників."""
        # Кнопки налаштувань запускають короткі HTTP/UDP дії у фонових worker-ах.
        self.wifi_button.clicked.connect(self._send_wifi_credentials)
        self.wifi_reset_button.clicked.connect(self._reset_wifi_credentials)
        self.discovery_button.clicked.connect(self._discover_robot)
        self.connect_button.clicked.connect(self._connect_robot)
        self.disconnect_button.clicked.connect(self._disconnect_robot)
        self.copy_log_button.clicked.connect(self._copy_log_to_clipboard)
        self.clear_log_button.clicked.connect(self._clear_log)
        self.speed_spin.editingFinished.connect(self._send_current_speed)
        self.robot_combo.currentIndexChanged.connect(self._select_discovered_robot)

        # Рух триває тільки поки кнопку утримують: press надсилає напрямок, release надсилає stop.
        self.forward_button.pressed.connect(lambda: self._start_motion("forward"))
        self.forward_button.released.connect(self._send_stop)
        self.backward_button.pressed.connect(lambda: self._start_motion("backward"))
        self.backward_button.released.connect(self._send_stop)
        self.left_button.pressed.connect(lambda: self._start_motion("left"))
        self.left_button.released.connect(self._send_stop)
        self.right_button.pressed.connect(lambda: self._start_motion("right"))
        self.right_button.released.connect(self._send_stop)
        self.stop_button.clicked.connect(self._send_stop)
        self.led_button.toggled.connect(self._toggle_led)

        # Автопілот / HSV-калібрування.
        self.autopilot_checkbox.toggled.connect(self._toggle_autopilot)
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        self.apply_hsv_button.clicked.connect(self._apply_hsv_from_ui)
        self.save_hsv_button.clicked.connect(self._save_hsv_to_config)
        self.mask_preview_checkbox.toggled.connect(self._on_autopilot_option_toggled)
        self.pin_mode_checkbox.toggled.connect(self._on_autopilot_option_toggled)

    # ----------------------------------------------------------------- підключення

    def _connect_robot(self) -> None:
        """Почати підключення до робота і перевірити WebSocket командою ping."""
        # Нове ручне підключення скасовує режим автоматичного перепідключення.
        self._manual_disconnect_requested = False
        self._reconnect_timer.stop()
        self._recovering_connection = False
        self._reconnect_in_flight = False
        try:
            # Старий клієнт закривається без stop, бо новий ping/stop виконається одразу нижче.
            self._close_robot_client(send_stop=False)
            self.robot_client = self._create_robot_client()
        except ValueError as exc:
            self._set_status(str(exc), level="ERROR")
            QMessageBox.warning(self, "Помилка параметрів", str(exc))
            return

        self._set_connected(False)
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)

        # Локальна змінна client фіксує саме той об’єкт, для якого стартував worker.
        client = self.robot_client
        self._set_status("Перевіряю WebSocket-з’єднання з ESP32-CAM...")

        def action() -> str:
            """Перевірити WebSocket і зупинити робота після підключення."""
            client.ping()
            client.stop()
            return "Підключення налаштовано. Натисніть і утримуйте кнопку руху."

        def on_success(message: object) -> None:
            """Оновити стан інтерфейсу після успішного підключення."""
            if self._manual_disconnect_requested or self.robot_client is not client:
                # Ігноруємо запізнілий результат, якщо користувач уже
                # відключився або створив інший клієнт.
                self._close_robot_client(send_stop=False)
                self._set_connected(False)
                return

            # Після успішного ping/stop усі transient-прапорці повертаються в початковий стан.
            self._disconnect_requested = False
            self._pending_stop = False
            self._stop_in_flight = False
            self._motion_command_in_flight = False
            self._led_in_flight = False
            self._speed_in_flight = False
            self._robot_error_count = 0
            self._set_connected(True)
            self._show_status_success(message)

            # Швидкість передається окремо, щоб прошивка мала актуальний currentSpeed.
            self._send_current_speed()
            QTimer.singleShot(800, self._start_video)

        def on_error(message: str) -> None:
            """Очистити клієнт і показати помилку підключення."""
            if self.robot_client is not client:
                return
            self._close_robot_client(send_stop=False)
            self._set_connected(False)
            self._set_status(f"Не вдалося підключитися до ESP32-CAM: {message}", level="ERROR")

        self._run_worker(action, on_success, on_error)

    def _disconnect_robot(self) -> None:
        """Почати ручне відключення з надсиланням stop і зупинкою відео."""
        # Ручне відключення вимикає автоматичне відновлення зв’язку.
        self._manual_disconnect_requested = True
        was_recovering = self._recovering_connection or self._reconnect_in_flight
        self._reconnect_timer.stop()
        self._recovering_connection = False
        self._reconnect_in_flight = False

        if was_recovering:
            self._close_robot_client(send_stop=False)
            self._stop_video()
            self._set_connected(False)
            self._set_status("Відключено.")
            return

        if self.robot_client is None:
            self._stop_video()
            self._set_connected(False)
            self._set_status("Відключено.")
            return

        self._pending_stop = False
        self._disconnect_requested = True

        # Stop відправляється до закриття клієнта, щоб робот не продовжив рух.
        self._send_stop()
        self._stop_video()
        if not self._has_active_robot_request() and not self._pending_stop:
            self._finalize_disconnect()
        else:
            self._set_connected(False)
            self._set_status(
                "Відключення: очікую завершення активної WebSocket-команди.",
            )

    # ------------------------------------------------------------------ Wi-Fi / discover

    def _send_wifi_credentials(self) -> None:
        """Запустити фонове передавання Wi-Fi налаштувань."""
        ssid = self.ssid_edit.text().strip()
        password = self.password_edit.text()
        setup_host = self.setup_host_edit.text().strip()
        port = self.command_port_spin.value()

        self._run_worker(
            lambda: self._send_wifi_action(setup_host, port, ssid, password),
            self._show_status_success,
            self._show_status_error,
        )

    def _send_wifi_action(
        self,
        setup_host: str,
        port: int,
        ssid: str,
        password: str,
    ) -> str:
        """Передати Wi-Fi налаштування через provisioning client."""
        client = WifiProvisioningClient(setup_host, port=port, timeout=WIFI_TIMEOUT_SECONDS)
        client.send_credentials(ssid, password)
        return "Wi-Fi налаштування передано. ESP32-CAM може перезавантажитися."

    def _discover_robot(self) -> None:
        """Запустити UDP-пошук роботів і оновити список в інтерфейсі."""
        self.discovery_button.setEnabled(False)
        self.robot_combo.clear()
        self._discovered_robots = []
        self._set_status("Пошук роботів ESP32-CAM у локальній мережі через UDP broadcast...")

        def on_success(result: object) -> None:
            """Заповнити список роботів після успішного UDP-пошуку."""
            self.discovery_button.setEnabled(True)
            robots = list(result or [])

            if not robots:
                self._set_status(
                    "Роботів не знайдено. Перевірте, що ПК і ESP32-CAM "
                    "перебувають у тій самій Wi-Fi мережі.",
                    level="WARNING",
                )
                return

            self._discovered_robots = robots
            self.robot_combo.blockSignals(True)
            self.robot_combo.clear()
            for robot in robots:
                self.robot_combo.addItem(robot.display_name)
            self.robot_combo.setCurrentIndex(0)
            self.robot_combo.blockSignals(False)

            # Перший знайдений робот одразу підставляється в поля для швидкого підключення.
            self._apply_discovered_robot(robots[0])

            if len(robots) == 1:
                self._set_status(f"Знайдено 1 робота: {robots[0].display_name}.")
            else:
                self._set_status(
                    f"Знайдено роботів: {len(robots)}. "
                    "Оберіть потрібного робота зі списку.",
                )

        def on_error(message: str) -> None:
            """Повернути кнопку пошуку і показати помилку discovery."""
            self.discovery_button.setEnabled(True)
            self._set_status(f"Помилка пошуку роботів: {message}", level="ERROR")

        self._run_worker(discover_robots, on_success, on_error)

    def _apply_discovered_robot(self, robot: RobotDiscoveryResult) -> None:
        """Заповнити поля підключення параметрами знайденого робота."""
        self.host_edit.setText(robot.ip)
        self.setup_host_edit.setText(robot.ip)
        self.command_port_spin.setValue(robot.websocket_port)
        self.stream_port_spin.setValue(robot.stream_port)

    def _select_discovered_robot(self, index: int) -> None:
        """Обробити вибір робота зі списку знайдених пристроїв."""
        if index < 0 or index >= len(self._discovered_robots):
            return
        robot = self._discovered_robots[index]
        self._apply_discovered_robot(robot)
        self._set_status(f"Обрано робота: {robot.display_name}.")

    def _reset_wifi_credentials(self) -> None:
        """Підтвердити і запустити скидання Wi-Fi налаштувань ESP32-CAM."""
        host = self.setup_host_edit.text().strip() or self.host_edit.text().strip()
        port = self.command_port_spin.value()

        if not host:
            QMessageBox.warning(
                self,
                "Скидання Wi-Fi",
                "Вкажіть IP ESP32-CAM або знайдіть робота автоматично.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Скидання Wi-Fi",
            "Скинути Wi-Fi налаштування ESP32-CAM? "
            "Після перезавантаження модуль запустить точку доступу "
            "KPI-Robot-Car-Setup.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.wifi_reset_button.setEnabled(False)

        def reset_action() -> str:
            """Виконати HTTP-запит скидання Wi-Fi налаштувань."""
            client = WifiProvisioningClient(host, port=port, timeout=WIFI_TIMEOUT_SECONDS)
            client.reset_credentials()
            return (
                "Wi-Fi налаштування скинуто. ESP32-CAM перезавантажується у setup-режим. "
                "Підключіться до Wi-Fi мережі KPI-Robot-Car-Setup і передайте "
                "нові SSID/пароль через форму."
            )

        def on_success(message: object) -> None:
            """Повернути кнопку скидання після успішного запиту."""
            self.wifi_reset_button.setEnabled(True)
            self._show_status_success(message)

        def on_error(message: str) -> None:
            """Повернути кнопку скидання після помилки."""
            self.wifi_reset_button.setEnabled(True)
            self._set_status(f"Не вдалося скинути Wi-Fi: {message}", level="ERROR")

        self._run_worker(reset_action, on_success, on_error)

    # ------------------------------------------------------------------ рух / speed

    def _start_motion(self, command: str) -> None:
        """Почати утримувану команду руху та запустити keep-alive."""
        if self._recovering_connection:
            self._set_status("Триває автоматичне перепідключення до ESP32-CAM.", level="WARNING")
            return
        if not self._connected or self.robot_client is None:
            self._set_status("Спочатку натисніть «Підключитися».", level="WARNING")
            return
        if self._disconnect_requested:
            self._set_status("Зараз виконується відключення від робота.", level="WARNING")
            return

        self._active_motion_command = command
        self._set_status(f"Утримується команда {command}. Відпустіть кнопку для stop.")
        self._send_motion(command, show_success=False)
        self._motion_keepalive_timer.start()

    def _repeat_motion_command(self) -> None:
        """Повторити активну команду руху для підтримки watchdog."""
        if self._active_motion_command is None:
            self._motion_keepalive_timer.stop()
            return
        self._send_motion(self._active_motion_command, show_success=False)

    def _send_motion(self, command: str, *, show_success: bool = True) -> None:
        """Надіслати команду руху у фоновому потоці."""
        if command == "stop":
            self._send_stop()
            return
        if not self._connected or self.robot_client is None or self._disconnect_requested:
            return

        if (
            self._motion_command_in_flight
            or self._stop_in_flight
            or self._pending_stop
            or self._led_in_flight
        ):
            return

        client = self.robot_client
        speed = self.speed_spin.value()
        self._motion_command_in_flight = True

        def action() -> str:
            """Надіслати одну команду руху з поточною швидкістю."""
            client.set_speed(speed)
            result = client.send_motion_command(command)
            response = f": {result.response_text}" if result.response_text else ""
            return f"Команда {command} виконана{response}"

        def on_success(message: object) -> None:
            """Обробити успішне виконання команди руху."""
            self._motion_command_in_flight = False
            self._record_robot_success()
            if show_success:
                self._show_status_success(message)
            self._after_robot_request_finished()

        def on_error(message: str) -> None:
            """Зупинити повторення руху після помилки команди."""
            self._motion_command_in_flight = False
            self._active_motion_command = None
            self._motion_keepalive_timer.stop()
            if self._record_robot_error(message):
                return
            self._after_robot_request_finished()

        self._run_worker(action, on_success, on_error)

    def _send_stop(self) -> None:
        """Надіслати stop або відкласти його до завершення активної команди."""
        self._active_motion_command = None
        self._motion_keepalive_timer.stop()
        if self.robot_client is None:
            if self._disconnect_requested:
                self._finalize_disconnect()
            return
        if self._stop_in_flight:
            return
        if self._motion_command_in_flight or self._led_in_flight or self._speed_in_flight:
            self._pending_stop = True
            return

        self._pending_stop = False
        self._stop_in_flight = True
        client = self.robot_client

        def action() -> str:
            """Виконати команду stop через поточний клієнт робота."""
            result = client.stop()
            response = f": {result.response_text}" if result.response_text else ""
            return f"Команда stop виконана{response}"

        def on_success(message: object) -> None:
            """Обробити успішне завершення stop-команди."""
            self._stop_in_flight = False
            self._record_robot_success()
            self._show_status_success(message)
            self._after_robot_request_finished()

        def on_error(message: str) -> None:
            """Обробити помилку stop-команди."""
            self._stop_in_flight = False
            if self._record_robot_error(message):
                return
            self._after_robot_request_finished()

        self._run_worker(action, on_success, on_error)

    def _send_current_speed(self) -> None:
        """Надіслати поточне значення швидкості, якщо робот готовий."""
        if (
            not self._connected
            or self.robot_client is None
            or self._disconnect_requested
            or self._speed_in_flight
            or self._has_active_robot_request()
        ):
            return

        client = self.robot_client
        speed = self.speed_spin.value()
        self._speed_in_flight = True

        def action() -> str:
            """Надіслати поточну швидкість до робота."""
            result = client.send_speed(speed)
            response = f": {result.response_text}" if result.response_text else ""
            return f"Швидкість встановлено {speed}{response}"

        def on_success(message: object) -> None:
            """Обробити успішне встановлення швидкості."""
            self._speed_in_flight = False
            self._record_robot_success()
            self._show_status_success(message)
            self._after_robot_request_finished()

        def on_error(message: str) -> None:
            """Показати помилку встановлення швидкості."""
            self._speed_in_flight = False
            self._set_status(f"Не вдалося встановити швидкість: {message}", level="WARNING")
            self._after_robot_request_finished()

        self._run_worker(action, on_success, on_error)

    def _toggle_led(self, checked: bool) -> None:
        """Увімкнути або вимкнути світлодіод через WebSocket."""
        if not self._connected or self.robot_client is None:
            self._set_led_checked(False)
            self._set_status("Спочатку натисніть «Підключитися».", level="WARNING")
            return
        if self._has_active_robot_request() or self._pending_stop:
            self._set_led_checked(not checked)
            self._set_status(
                "WebSocket-команда ще виконується. "
                "Повторіть перемикання світлодіода пізніше.",
                level="WARNING",
            )
            return

        client = self.robot_client
        self._set_led_checked(checked)
        self.led_button.setEnabled(False)
        self._led_in_flight = True

        def action() -> str:
            """Виконати команду LED для переданого стану."""
            result = client.led_on() if checked else client.led_off()
            response = f": {result.response_text}" if result.response_text else ""
            return f"Світлодіод {'увімкнено' if checked else 'вимкнено'}{response}"

        def on_success(message: object) -> None:
            """Оновити інтерфейс після успішної LED-команди."""
            self._led_in_flight = False
            self._record_robot_success()
            self._set_led_checked(checked)
            self.led_button.setEnabled(self._connected)
            self._show_status_success(message)
            self._after_robot_request_finished()

        def on_error(message: str) -> None:
            """Відкотити стан кнопки LED після помилки."""
            self._led_in_flight = False
            self._set_led_checked(not checked)
            self.led_button.setEnabled(self._connected)
            self._set_status(
                f"Не вдалося перемкнути світлодіод: {message}",
                level="WARNING",
            )
            self._after_robot_request_finished()

        self._run_worker(action, on_success, on_error)

    def _send_stop_blocking(self) -> None:
        """Синхронно надіслати stop під час закриття вікна."""
        if self.robot_client is None or self._has_active_robot_request():
            return
        try:
            self.robot_client.stop()
        except Exception:  # noqa: BLE001
            return

    # ------------------------------------------------------------------ відеопотік

    def _start_video(self) -> None:
        """Запустити фоновий MJPEG-потік, якщо підключення активне."""
        if not self._connected or self._disconnect_requested:
            return

        self._stop_video()
        if self._video_thread is not None:
            self._set_status(
                "Попередній відеопотік ще завершується. Новий потік не запущено.",
                level="WARNING",
            )
            return

        self._video_thread = VideoStreamThread(
            host=self.host_edit.text().strip(),
            stream_port=self.stream_port_spin.value(),
        )
        self._video_thread.frame_ready.connect(self._update_video_frame)
        self._video_thread.status_changed.connect(self._set_status)

        # Якщо автопілот уже активний — підписуємо його на сирі кадри.
        if self._autopilot is not None:
            self._video_thread.cv_frame_ready.connect(self._autopilot.on_frame)

        self._video_thread.start()

    def _stop_video(self) -> None:
        """Зупинити MJPEG-потік і повернути placeholder відео."""
        if self._video_thread is not None:
            # Відписуємо автопілот, якщо він прив'язаний.
            if self._autopilot is not None:
                try:
                    self._video_thread.cv_frame_ready.disconnect(self._autopilot.on_frame)
                except TypeError:
                    pass
            stopped = self._video_thread.stop()
            if stopped:
                self._video_thread = None
            else:
                self._set_status(
                    "Відеопотік не завершився коректно. "
                    "Потік залишено активним до завершення.",
                    level="WARNING",
                )

        # Після stop завжди очищаємо QLabel.
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(VIDEO_PLACEHOLDER_TEXT)

    def _update_video_frame(self, image: QImage) -> None:
        """Показати отриманий кадр у QLabel з масштабуванням."""
        pixmap = self._draw_autopilot_overlay(QPixmap.fromImage(image))
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.video_label.setPixmap(scaled)

    # ------------------------------------------------------------------ створення клієнта

    def _create_robot_client(self) -> RobotClient:
        """Створити RobotClient із поточних полів інтерфейсу."""
        client = RobotClient(
            self.host_edit.text().strip(),
            command_port=self.command_port_spin.value(),
        )
        client.set_speed(self.speed_spin.value())
        return client

    def _has_active_robot_request(self) -> bool:
        """Перевірити, чи зараз виконується будь-який запит до робота."""
        return (
            self._motion_command_in_flight
            or self._stop_in_flight
            or self._led_in_flight
            or self._speed_in_flight
            or self._reconnect_in_flight
        )

    def _record_robot_success(self) -> None:
        """Скинути лічильник послідовних помилок робота."""
        self._robot_error_count = 0

    def _record_robot_error(self, message: str) -> bool:
        """Зафіксувати помилку робота і запустити відновлення за порогом."""
        self._robot_error_count += 1
        self._show_status_error(message)
        if self._robot_error_count >= self._max_robot_errors:
            self._handle_robot_connection_lost()
            return True
        return False

    def _handle_robot_connection_lost(self) -> None:
        """Обробити втрату зв’язку та за потреби запустити перепідключення."""
        self._motion_keepalive_timer.stop()
        self._active_motion_command = None
        self._pending_stop = False
        self._motion_command_in_flight = False
        self._stop_in_flight = False
        self._led_in_flight = False
        self._speed_in_flight = False
        self._reconnect_in_flight = False
        self._disconnect_requested = False
        self._robot_error_count = 0

        self._stop_video()
        self._close_robot_client(send_stop=False)

        if ROBOT_AUTO_RECONNECT_ENABLED and not self._manual_disconnect_requested:
            self._recovering_connection = True
            self._reconnect_attempts = 0
            self._set_connected(False)
            self._set_status(
                "Зв’язок із ESP32-CAM тимчасово втрачено. Виконую автоматичне перепідключення...",
                level="WARNING",
            )
            self._reconnect_timer.start()
            return

        self._recovering_connection = False
        self._set_connected(False)
        self._set_status(
            "Зв’язок із ESP32-CAM втрачено. Натисніть «Підключитися» повторно.",
            level="ERROR",
        )

    def _try_auto_reconnect(self) -> None:
        """Виконати одну спробу автоматичного перепідключення."""
        if self._manual_disconnect_requested:
            self._reconnect_timer.stop()
            self._recovering_connection = False
            self._reconnect_in_flight = False
            self._set_connected(False)
            return

        if self.robot_client is not None or self._has_active_robot_request():
            return

        self._reconnect_attempts += 1
        self._reconnect_in_flight = True
        self._set_status(f"Спроба автоматичного перепідключення #{self._reconnect_attempts}...")

        try:
            self.robot_client = self._create_robot_client()
        except ValueError as exc:
            self._reconnect_timer.stop()
            self._recovering_connection = False
            self._reconnect_in_flight = False
            self._set_connected(False)
            self._set_status(str(exc), level="ERROR")
            return

        client = self.robot_client

        def action() -> str:
            """Перевірити WebSocket під час автоматичного перепідключення."""
            client.ping()
            client.stop()
            return "Підключення до ESP32-CAM відновлено."

        def on_success(message: object) -> None:
            """Відновити стан GUI після успішного перепідключення."""
            if self._manual_disconnect_requested or self.robot_client is not client:
                self._close_robot_client(send_stop=False)
                self._set_connected(False)
                return

            self._reconnect_timer.stop()
            self._recovering_connection = False
            self._reconnect_in_flight = False
            self._robot_error_count = 0
            self._disconnect_requested = False
            self._pending_stop = False
            self._set_connected(True)
            self._show_status_success(message)

            self._send_current_speed()
            QTimer.singleShot(800, self._start_video)

        def on_error(message: str) -> None:
            """Підготувати наступну спробу після помилки перепідключення."""
            if self.robot_client is not client:
                self._reconnect_in_flight = False
                return
            self._reconnect_in_flight = False
            self._close_robot_client(send_stop=False)
            self._set_connected(False)
            self._set_status(
                f"Автоматичне перепідключення не вдалося: {message}",
                level="WARNING",
            )

        self._run_worker(action, on_success, on_error)

    def _after_robot_request_finished(self) -> None:
        """Запустити відкладений stop або фіналізацію відключення."""
        if self._pending_stop and not self._has_active_robot_request():
            self._send_stop()
            return
        if self._disconnect_requested and not self._has_active_robot_request():
            self._finalize_disconnect()

    def _finalize_disconnect(self) -> None:
        """Завершити стан ручного відключення після всіх команд."""
        self._pending_stop = False
        self._disconnect_requested = False
        self._close_robot_client(send_stop=False)
        self._set_connected(False)
        self._set_status(
            "Відключено. Команду stop надіслано, якщо ESP32-CAM доступна.",
        )

    def _close_robot_client(self, *, send_stop: bool = True) -> None:
        """Закрити поточного RobotClient і очистити посилання."""
        if self.robot_client is not None:
            self.robot_client.close(send_stop=send_stop)
            self.robot_client = None

    # ------------------------------------------------------------------ worker helper

    def _run_worker(
        self,
        action: Callable[[], Any],
        on_success: Callable[[object], None],
        on_error: Callable[[str], None],
    ) -> None:
        """Запустити дію в QThreadPool і під’єднати обробники результату."""
        worker = CommandWorker(action)
        worker.signals.success.connect(on_success)
        worker.signals.error.connect(on_error)
        self._thread_pool.start(worker)

    # ------------------------------------------------------------------ статус / лог

    def _show_status_success(self, message: object) -> None:
        """Показати успішне статусне повідомлення."""
        self._set_status(str(message))

    def _show_status_error(self, message: str) -> None:
        """Показати статусне повідомлення про помилку."""
        self._set_status(message, level="ERROR")

    def _set_status(self, message: str, level: str = "INFO") -> None:
        """Оновити status bar і додати запис у журнал без дублювання."""
        clean_message = " ".join(str(message).split())
        if not clean_message:
            return
        level = level.upper()
        self.statusBar().showMessage(clean_message)
        log_key = (level, clean_message)
        if log_key == self._last_log_entry:
            return
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_edit.appendPlainText(f"{timestamp} [{level:<7}] {clean_message}")
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )
        self._last_log_entry = log_key

    def _copy_log_to_clipboard(self) -> None:
        """Скопіювати весь журнал статусів у буфер обміну."""
        QApplication.clipboard().setText(self.log_edit.toPlainText())
        self._set_status("Журнал скопійовано в буфер обміну.")

    def _clear_log(self) -> None:
        """Очистити журнал статусів і скинути останній запис."""
        self.log_edit.clear()
        self._last_log_entry = None
        self._set_status("Журнал очищено.")

    def _set_led_checked(self, checked: bool) -> None:
        """Оновити позначення LED-кнопки без запуску обробника toggled."""
        self.led_button.blockSignals(True)
        self.led_button.setChecked(checked)
        self.led_button.setText(f"Світлодіод: {'увімкнено' if checked else 'вимкнено'}")
        self.led_button.blockSignals(False)

    def _set_connected(self, connected: bool) -> None:
        """Оновити доступність кнопок відповідно до стану підключення."""
        self._connected = connected
        self.connect_button.setEnabled(not connected and not self._recovering_connection)
        self.disconnect_button.setEnabled(connected or self._recovering_connection)

        for button in (
            self.forward_button,
            self.backward_button,
            self.left_button,
            self.right_button,
        ):
            # У режимі автопілота ручні кнопки блокуються окремо.
            button.setEnabled(connected and not self._autopilot_enabled)

        self.led_button.setEnabled(connected)
        if not connected:
            self._set_led_checked(False)
        self.stop_button.setEnabled(True)

    # ---------------------------------------------------------- HSV / автопілот helpers

    def _load_hsv_to_ui(self, color_name: str) -> None:
        """Підставити HSV-діапазон вибраного кольору у спінбокси."""
        rng = self._color_ranges.get(color_name)
        if rng is None:
            return
        h_min, s_min, v_min = rng.lower
        h_max, s_max, v_max = rng.upper

        self.h_min_spin.blockSignals(True)
        self.h_max_spin.blockSignals(True)
        self.s_min_spin.blockSignals(True)
        self.s_max_spin.blockSignals(True)
        self.v_min_spin.blockSignals(True)
        self.v_max_spin.blockSignals(True)

        self.h_min_spin.setValue(h_min)
        self.h_max_spin.setValue(h_max)
        self.s_min_spin.setValue(s_min)
        self.s_max_spin.setValue(s_max)
        self.v_min_spin.setValue(v_min)
        self.v_max_spin.setValue(v_max)

        self.h_min_spin.blockSignals(False)
        self.h_max_spin.blockSignals(False)
        self.s_min_spin.blockSignals(False)
        self.s_max_spin.blockSignals(False)
        self.v_min_spin.blockSignals(False)
        self.v_max_spin.blockSignals(False)

    def _on_color_changed(self, name: str) -> None:
        """Оновити UI при зміні вибраного кольору."""
        if not name:
            return
        self._current_color_name = name
        self._load_hsv_to_ui(name)
        if self._autopilot is not None:
            cfg = self._build_autopilot_config_from_ui()
            self._autopilot.update_config(cfg)
        self._set_status(f"Обрано колір автопілота: {name}.")

    def _read_hsv_from_ui(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Прочитати HSV-діапазон безпосередньо зі спінбоксів."""
        lower = (
            self.h_min_spin.value(),
            self.s_min_spin.value(),
            self.v_min_spin.value(),
        )
        upper = (
            self.h_max_spin.value(),
            self.s_max_spin.value(),
            self.v_max_spin.value(),
        )
        return lower, upper

    def _apply_hsv_from_ui(self) -> None:
        """Оновити діапазон для поточного кольору з UI та оновити конфіг автопілота."""
        name = self._current_color_name
        lower, upper = self._read_hsv_from_ui()
        self._color_ranges[name] = ColorRange(name=name, lower=lower, upper=upper)
        self._set_status(
            f"Оновлено HSV-діапазон для кольору {name}: "
            f"H[{lower[0]}..{upper[0]}], S[{lower[1]}..{upper[1]}], V[{lower[2]}..{upper[2]}]."
        )

        if self._autopilot is not None:
            cfg = self._build_autopilot_config_from_ui()
            self._autopilot.update_config(cfg)

    def _save_hsv_to_config(self) -> None:
        """Зберегти усі поточні HSV-діапазони в JSON-конфіг."""
        self._apply_hsv_from_ui()
        save_color_ranges(self._color_ranges.values())
        self._set_status("HSV-діапазони збережено у config/hsv_ranges.json.")

    def _build_autopilot_config_from_ui(self) -> AutoPilotConfig:
        """Побудувати AutoPilotConfig на основі поточних UI-налаштувань."""
        lower, upper = self._read_hsv_from_ui()

        secondary_lower: tuple[int, int, int] | None = None
        secondary_upper: tuple[int, int, int] | None = None
        if self._current_color_name == "red":
            red2 = self._color_ranges.get("red2")
            if red2 is not None:
                secondary_lower = red2.lower
                secondary_upper = red2.upper

        detector_enabled = self.detector_checkbox.isChecked()
        classes_raw = self.detector_classes_edit.text().strip()
        target_classes = [
            cls.strip().lower()
            for cls in classes_raw.split(",")
            if cls.strip()
        ]

        cfg = AutoPilotConfig(
            lower_hsv=lower,
            upper_hsv=upper,
            secondary_lower_hsv=secondary_lower,
            secondary_upper_hsv=secondary_upper,
            shape_filter_enabled=self.shape_filter_checkbox.isChecked(),
            detector_enabled=detector_enabled,
            detector_target_classes=tuple(target_classes),
            mask_preview_enabled=self.mask_preview_checkbox.isChecked(),
            pin_mode_enabled=self.pin_mode_checkbox.isChecked(),
        )
        return cfg

    def _toggle_autopilot(self, enabled: bool) -> None:
        """Увімкнути/вимкнути автопілот."""
        if enabled:
            if not self._connected or self.robot_client is None:
                self._set_status("Спочатку підключіться до робота.", level="WARNING")
                self.autopilot_checkbox.blockSignals(True)
                self.autopilot_checkbox.setChecked(False)
                self.autopilot_checkbox.blockSignals(False)
                return
            self._start_autopilot()
        else:
            self._stop_autopilot()

    def _start_autopilot(self) -> None:
        """Створити та запустити автопілот, підписавши його на відеопотік."""
        if self._autopilot is not None:
            return

        self._apply_hsv_from_ui()
        cfg = self._build_autopilot_config_from_ui()
        if cfg.detector_enabled:
            self._set_status(
                "Увага: YOLO увімкнено — робот їде лише якщо знайдено і колір, і клас YOLO. "
                "Для слідкування за кольором вимкніть YOLO.",
                level="WARNING",
            )
        self._autopilot = ColorFollowAutoPilot(cfg, self)
        self._autopilot.command_ready.connect(self._on_autopilot_command)
        self._autopilot.target_detected.connect(self._on_autopilot_target)
        self._autopilot.mask_ready.connect(self._on_autopilot_mask)
        self._autopilot.mission_status.connect(self._on_autopilot_mission_status)
        self._autopilot.mission_complete.connect(self._on_autopilot_mission_complete)
        self._autopilot.start()

        if self._video_thread is not None:
            self._video_thread.cv_frame_ready.connect(self._autopilot.on_frame)

        for button in (
            self.forward_button,
            self.backward_button,
            self.left_button,
            self.right_button,
        ):
            button.setEnabled(False)

        self._autopilot_enabled = True
        self._set_status("Автопілот увімкнено.")

    def _stop_autopilot(self) -> None:
        """Зупинити автопілот і повернути ручне керування."""
        if self._autopilot is None:
            return

        if self._video_thread is not None:
            try:
                self._video_thread.cv_frame_ready.disconnect(self._autopilot.on_frame)
            except TypeError:
                pass

        try:
            self._autopilot.command_ready.disconnect(self._on_autopilot_command)
        except TypeError:
            pass
        try:
            self._autopilot.target_detected.disconnect(self._on_autopilot_target)
        except TypeError:
            pass
        try:
            self._autopilot.mask_ready.disconnect(self._on_autopilot_mask)
        except TypeError:
            pass
        try:
            self._autopilot.mission_status.disconnect(self._on_autopilot_mission_status)
        except TypeError:
            pass
        try:
            self._autopilot.mission_complete.disconnect(self._on_autopilot_mission_complete)
        except TypeError:
            pass

        self._autopilot.stop()
        self._autopilot_target = None
        self._autopilot_mask = None
        self._autopilot.deleteLater()
        self._autopilot = None

        for button in (
            self.forward_button,
            self.backward_button,
            self.left_button,
            self.right_button,
        ):
            button.setEnabled(self._connected)

        self._autopilot_enabled = False
        self._set_status("Автопілот вимкнено.")

    def _on_autopilot_target(self, target: object) -> None:
        """Зберегти останню знайдену ціль для відображення на відео."""
        self._autopilot_target = target if isinstance(target, dict) else None

    def _on_autopilot_mask(self, mask: object) -> None:
        """Зберегти останню маску кольору для накладання на відео."""
        self._autopilot_mask = mask

    def _on_autopilot_option_toggled(self, _enabled: bool) -> None:
        """Оновити конфіг автопілота при зміні додаткових опцій."""
        if self._autopilot is not None:
            self._autopilot.update_config(self._build_autopilot_config_from_ui())

    def _on_autopilot_mission_status(self, message: str) -> None:
        """Показати стан місії збивання кегель у журналі."""
        self._set_status(message)

    def _on_autopilot_mission_complete(self) -> None:
        """Зупинити автопілот після збиття всіх кегель обраного кольору."""
        self._set_status(
            "Усі кеглі потрібного кольору збито. Автопілот зупинено.",
            level="INFO",
        )
        if self.autopilot_checkbox.isChecked():
            self.autopilot_checkbox.setChecked(False)

    def _on_autopilot_command(self, command: str) -> None:
        """Прийняти команду руху від автопілота та відправити її роботу."""
        if not self._autopilot_enabled:
            return
        if not self._connected or self.robot_client is None:
            return

        if command == "stop":
            if self._active_motion_command is not None:
                self._send_stop()
            return

        if command == self._active_motion_command:
            return

        self._active_motion_command = command
        self._send_motion(command, show_success=False)
        self._motion_keepalive_timer.start()

    def _draw_autopilot_overlay(self, pixmap: QPixmap) -> QPixmap:
        """Намалювати маску кольору, рамку цілі та лінію центру на кадрі відео."""
        if not self._autopilot_enabled:
            return pixmap

        import numpy as np

        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        bytes_per_line = image.bytesPerLine()
        ptr = image.bits()
        ptr.setsize(height * bytes_per_line)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
        rgb = arr[:, : width * 3].reshape((height, width, 3)).copy()

        if self.mask_preview_checkbox.isChecked() and self._autopilot_mask is not None:
            mask = self._autopilot_mask
            if isinstance(mask, np.ndarray) and mask.shape == (height, width):
                selected = mask > 0
                rgb[selected, 0] = np.clip(rgb[selected, 0] * 0.35 + 255 * 0.65, 0, 255)
                rgb[selected, 1] = (rgb[selected, 1] * 0.35).astype(np.uint8)
                rgb[selected, 2] = (rgb[selected, 2] * 0.35).astype(np.uint8)

        overlay = QPixmap.fromImage(
            QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888).copy()
        )
        painter = QPainter(overlay)

        if self._autopilot_target is not None:
            target = self._autopilot_target
            pin_state = target.get("pin_state", "")
            if pin_state == "ram":
                color = QColor(255, 80, 80)
            elif target.get("confirmed", True):
                color = QColor(0, 255, 0)
            else:
                color = QColor(255, 165, 0)
            pen = QPen(color)
            pen.setWidth(max(2, overlay.width() // 200))
            painter.setPen(pen)
            painter.drawRect(target["x"], target["y"], target["w"], target["h"])

            center_pen = QPen(QColor(255, 255, 0))
            center_pen.setWidth(max(1, overlay.width() // 320))
            painter.setPen(center_pen)
            frame_center = overlay.width() // 2
            painter.drawLine(frame_center, 0, frame_center, overlay.height())

        painter.end()
        return overlay
