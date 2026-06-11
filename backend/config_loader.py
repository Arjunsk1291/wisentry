"""Configuration loader for WiSentry.

Reads config.yaml from the project root, validates that every required
section exists, and hands the rest of the system a plain nested dict.

Runs on: any OS with Python 3.9+. Dependencies: PyYAML.
"""

from pathlib import Path

import yaml

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT_DIRECTORY / "config.yaml"
REQUIRED_SECTIONS = [
    "network",
    "dashboard",
    "room",
    "signal",
    "detection",
    "ml",
    "logging",
    "simulation",
]


class ConfigError(Exception):
    """Raised when config.yaml is missing, unreadable, or incomplete."""


def load_config(config_path=None):
    """Load and validate the WiSentry configuration file.

    Args:
        config_path (str | Path | None): Path to a YAML config file. When
            None, the project-root config.yaml is used.

    Returns:
        dict: The parsed configuration with all REQUIRED_SECTIONS present.

    Raises:
        ConfigError: If the file is missing, not valid YAML, or lacks a
            required section.
    """
    resolved_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not resolved_path.exists():
        raise ConfigError(
            f"Configuration file not found: {resolved_path}. "
            "Copy config.yaml into the project root or pass --config."
        )
    try:
        with open(resolved_path, "r", encoding="utf-8") as config_file:
            parsed_config = yaml.safe_load(config_file)
    except yaml.YAMLError as yaml_error:
        raise ConfigError(
            f"config.yaml is not valid YAML: {yaml_error}"
        ) from yaml_error

    if not isinstance(parsed_config, dict):
        raise ConfigError("config.yaml must contain a top-level mapping.")

    missing_sections = [
        section for section in REQUIRED_SECTIONS if section not in parsed_config
    ]
    if missing_sections:
        raise ConfigError(
            "config.yaml is missing required sections: "
            + ", ".join(missing_sections)
        )
    return parsed_config


if __name__ == "__main__":
    loaded_configuration = load_config()
    print("config_loader self-test:")
    for section_name in REQUIRED_SECTIONS:
        section_key_count = len(loaded_configuration[section_name])
        print(f"  section '{section_name}': {section_key_count} keys")
    print("PASS — configuration loaded and validated.")
