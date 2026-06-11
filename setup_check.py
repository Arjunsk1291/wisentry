"""WiSentry environment verifier.

Run this before the first start (and after any environment change):

    python setup_check.py

It verifies the Python version, every third-party dependency, the presence
and validity of config.yaml, and that the UDP and dashboard ports are free.
Exits 0 when the environment is ready, 1 otherwise.

Runs on: Windows 10/11, Ubuntu 20.04+. Dependencies: only the standard
library plus the packages it is checking for.
"""

import importlib
import socket
import sys
from pathlib import Path

MINIMUM_PYTHON_VERSION = (3, 9)
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PyYAML", "yaml"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("dash", "dash"),
    ("plotly", "plotly"),
    ("pytest", "pytest"),
]
PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent
CONFIG_FILE_PATH = PROJECT_ROOT_DIRECTORY / "config.yaml"

PASS_MARK = "[ OK ]"
FAIL_MARK = "[FAIL]"


def check_python_version() -> bool:
    """Verify the interpreter is at least MINIMUM_PYTHON_VERSION.

    Returns:
        bool: True when the running Python version is new enough.
    """
    current_version = sys.version_info[:2]
    if current_version >= MINIMUM_PYTHON_VERSION:
        print(f"{PASS_MARK} Python {sys.version.split()[0]}")
        return True
    print(
        f"{FAIL_MARK} Python {current_version[0]}.{current_version[1]} found; "
        f"need {MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1]}+. "
        "Install a newer Python from https://python.org and retry."
    )
    return False


def check_required_packages() -> bool:
    """Try importing every third-party dependency WiSentry needs.

    Returns:
        bool: True when every package imports successfully.
    """
    all_packages_present = True
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            imported_module = importlib.import_module(import_name)
            package_version = getattr(imported_module, "__version__", "unknown")
            print(f"{PASS_MARK} {pip_name} {package_version}")
        except ImportError:
            print(
                f"{FAIL_MARK} {pip_name} is not installed. "
                f"Fix with: pip install -r requirements.txt"
            )
            all_packages_present = False
    return all_packages_present


def check_config_file() -> bool:
    """Verify config.yaml exists and parses as valid YAML.

    Returns:
        bool: True when the configuration file loads cleanly.
    """
    if not CONFIG_FILE_PATH.exists():
        print(f"{FAIL_MARK} config.yaml not found at {CONFIG_FILE_PATH}")
        return False
    try:
        import yaml

        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as config_file:
            parsed_config = yaml.safe_load(config_file)
        if not isinstance(parsed_config, dict):
            print(f"{FAIL_MARK} config.yaml parsed but is not a mapping.")
            return False
        print(f"{PASS_MARK} config.yaml valid ({len(parsed_config)} sections)")
        return True
    except ImportError:
        print(f"{FAIL_MARK} PyYAML missing; cannot validate config.yaml.")
        return False
    except Exception as parse_error:  # yaml.YAMLError plus IO errors
        print(f"{FAIL_MARK} config.yaml is invalid YAML: {parse_error}")
        return False


def check_port_is_free(port_number: int, protocol_name: str) -> bool:
    """Check that a TCP/UDP port can be bound on localhost.

    Args:
        port_number (int): The port to test.
        protocol_name (str): "udp" or "tcp" — selects the socket type.

    Returns:
        bool: True when the port could be bound (i.e. it is free).
    """
    socket_type = socket.SOCK_DGRAM if protocol_name == "udp" else socket.SOCK_STREAM
    test_socket = socket.socket(socket.AF_INET, socket_type)
    try:
        test_socket.bind(("127.0.0.1", port_number))
        print(f"{PASS_MARK} {protocol_name.upper()} port {port_number} is free")
        return True
    except OSError:
        print(
            f"{FAIL_MARK} {protocol_name.upper()} port {port_number} is busy. "
            "Another program (or a previous WiSentry run) is using it — "
            "close it or change the port in config.yaml."
        )
        return False
    finally:
        test_socket.close()


def run_all_checks() -> int:
    """Run every environment check and report a summary.

    Returns:
        int: 0 when every check passed, 1 otherwise.
    """
    print("WiSentry environment check\n" + "-" * 40)
    check_results = [
        check_python_version(),
        check_required_packages(),
        check_config_file(),
        check_port_is_free(5566, "udp"),
        check_port_is_free(8050, "tcp"),
    ]
    print("-" * 40)
    if all(check_results):
        print("All checks passed. You are ready: python main.py --simulate")
        return 0
    print("Some checks FAILED — fix the items above and run this again.")
    return 1


if __name__ == "__main__":
    sys.exit(run_all_checks())
