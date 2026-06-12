"""Центральні налаштування контролера ESP32-CAM робота."""

# Порти та URL-шляхи мають збігатися з HTTP/WebSocket handlers у прошивці.
DEFAULT_WS_PORT = 80
DEFAULT_STREAM_PORT = 81
DEFAULT_STREAM_PATH = "/stream"
DEFAULT_WS_PATH = "/ws"
DEFAULT_ACTION_PATH = "/action"
DEFAULT_WIFI_PATH = "/wifi"
DEFAULT_WIFI_RESET_PATH = "/wifi/reset"

# DEFAULT_SETUP_HOST використовується для setup-точки доступу після скидання Wi-Fi.
# DEFAULT_HOST є початковою адресою робота у графічному інтерфейсі.
DEFAULT_SETUP_HOST = "192.168.4.1"
DEFAULT_HOST = "192.168.4.1"

# UDP discovery приймає broadcast-повідомлення з таким підписом на цьому порту.
DISCOVERY_PORT = 4210
DISCOVERY_TIMEOUT_SECONDS = 5.0
DISCOVERY_SIGNATURE = "KPI_ROBOT_CAR"

# Обмеження PWM-швидкості повторюють допустимий діапазон прошивки.
DEFAULT_SPEED = 170
MIN_SPEED = 85
MAX_SPEED = 255

# Timeout-и короткі, щоб GUI не зависав на недоступному роботі.
WS_CONNECT_TIMEOUT_SECONDS = 1.5
WS_RESPONSE_TIMEOUT_SECONDS = 1.5
WIFI_TIMEOUT_SECONDS = 3.0
VIDEO_RECONNECT_DELAY_SECONDS = 1.0

# Watchdog і keep-alive визначають, як довго робот рухається без нової команди.
MOTION_WATCHDOG_MS = 500
MOTION_KEEPALIVE_INTERVAL_MS = 300

# Швидкості автопілота: повільніше сканування, швидший таран і від'їзд після збиття.
AUTOPILOT_SCAN_SPEED = 130
AUTOPILOT_HIT_SPEED = 240
AUTOPILOT_BACKUP_SPEED = 220

# Поріг помилок і таймер перепідключення керують автоматичним відновленням зв’язку.
ROBOT_ERROR_THRESHOLD = 5
ROBOT_RECONNECT_INTERVAL_MS = 2000
ROBOT_AUTO_RECONNECT_ENABLED = True

# Тексти інтерфейсу зібрані тут, щоб не дублювати сталі значення у GUI.
APP_NAME = "KPI Robot Vision Car"
VIDEO_PLACEHOLDER_TEXT = "Відеопотік не запущено"

# ---------------------------------------------------------------------------
# Комп'ютерний зір: дефолтні HSV-діапазони та файл користувацької конфігурації
# ---------------------------------------------------------------------------

# Файл збережених HSV-діапазонів (JSON) зберігається у каталозі config/.
HSV_CONFIG_FILENAME = "hsv_ranges.json"

# Діапазони підібрані так, щоб чітко розділяти відтінки:
# червоний/оранжевий/жовтий, синій/блакитний, рожевий/фіолетовий тощо.
# H: 0–179, S: 0–255, V: 0–255.
DEFAULT_COLOR_HSV_RANGES: dict[str, dict[str, tuple[int, int, int]]] = {
    # Червоно-рожевий сектор розбитий на піддіапазони для кращої роздільної здатності.
    "red": {"lower": (0, 100, 60), "upper": (10, 255, 255)},
    "red2": {"lower": (170, 100, 60), "upper": (179, 255, 255)},
    "orange": {"lower": (8, 100, 60), "upper": (25, 255, 255)},
    # Жовтий: низька S/V мін — для блідого жовтого на ESP32-CAM
    "yellow": {"lower": (10, 25, 110), "upper": (42, 255, 255)},
    "yellow_green": {"lower": (36, 90, 60), "upper": (50, 255, 255)},

    # Зелений + бірюзовий
    "green": {"lower": (40, 80, 50), "upper": (85, 255, 255)},
    "cyan": {"lower": (81, 80, 50), "upper": (95, 255, 255)},

    # Синій / блакитний (H < 125, щоб не перетинатися з violet)
    "blue": {"lower": (103, 65, 35), "upper": (124, 255, 255)},
    "light_blue": {"lower": (88, 35, 90), "upper": (102, 255, 255)},

    # Фіолетовий / рожевий / магента
    "violet": {"lower": (125, 80, 50), "upper": (145, 255, 255)},
    "magenta": {"lower": (140, 90, 60), "upper": (165, 255, 255)},
    "pink": {"lower": (155, 60, 100), "upper": (170, 255, 255)},
}
