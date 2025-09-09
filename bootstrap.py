import os
import sys
import logging
import subprocess
import importlib.util
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configures logging to a rotating file."""
    if os.name == 'nt':  # Windows
        log_dir = Path(os.getenv('APPDATA', '')) / 'SpriteForge' / 'logs'
    else:  # Linux, macOS
        log_dir = Path.home() / '.sprite_forge' / 'logs'

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'sprite_forge.log'

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
    handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    logging.info("Logging initialized from bootstrap.")

def check_dependencies():
    """Checks for required dependencies and prompts for installation if missing."""
    required = {"PIL": "Pillow", "PyQt6": "PyQt6", "numpy": "numpy", "matplotlib": "matplotlib"}
    missing_imports = [name for name in required.keys() if importlib.util.find_spec(name) is None]
    if not missing_imports:
        return True

    packages_to_install = [required[name] for name in missing_imports]
    install_command = f"pip install {' '.join(packages_to_install)}"
    is_headless = any(arg in sys.argv for arg in ['--batch', '--selftest', '--headless'])

    if "PyQt6" in missing_imports or is_headless:
        print(f"Error: Missing required dependencies: {', '.join(packages_to_install)}", file=sys.stderr)
        if is_headless:
            print(f"Please run command: {install_command}", file=sys.stderr)
        else:
            if input("Install now? [y/N] ").lower() == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
                print("Complete. Please restart.")
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication, QMessageBox
    app = QApplication(sys.argv)
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setText(f"Missing: {', '.join(packages_to_install)}")
    msg_box.setInformativeText("Install them now?")
    msg_box.setDetailedText(install_command)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    if msg_box.exec() == QMessageBox.StandardButton.Yes:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
            QMessageBox.information(None, "Success", "Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            QMessageBox.critical(None, "Failed", str(e))
    sys.exit(1)
