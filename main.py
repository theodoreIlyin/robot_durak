"""Точка входу графічного застосунку KPI Robot Vision Car."""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from config.settings import APP_NAME
from gui.main_window import MainWindow


def main() -> int:
    """Запустити Qt-застосунок і показати головне вікно."""

    # QApplication повинен існувати до створення будь-яких Qt-віджетів.
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    # MainWindow інкапсулює весь стан підключення, відео та ручного керування.
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    # SystemExit повертає код завершення Qt event loop у процес.
    raise SystemExit(main())
