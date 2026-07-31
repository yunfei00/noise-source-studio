"""Configuration and platform path tests."""

import json

from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.infrastructure.config import SettingsManager


def test_configuration_directories_are_created(application_paths: ApplicationPaths) -> None:
    manager = SettingsManager(application_paths)

    settings = manager.load()

    assert application_paths.config_file.is_file()
    assert settings.output_directory.is_dir()
    assert settings.model_directory.is_dir()
    assert settings.log_directory.is_dir()


def test_configuration_round_trip(application_paths: ApplicationPaths) -> None:
    manager = SettingsManager(application_paths)
    settings = manager.load().model_copy(
        update={"device_preference": "cpu", "allow_cpu_fallback": False}
    )

    manager.save(settings)

    restored = manager.load()
    assert restored.device_preference == "cpu"
    assert not restored.allow_cpu_fallback


def test_legacy_default_device_is_migrated_and_cpu_survives_restart(
    application_paths: ApplicationPaths,
) -> None:
    manager = SettingsManager(application_paths)
    manager.load()
    payload = json.loads(application_paths.config_file.read_text(encoding="utf-8"))
    payload.pop("device_preference", None)
    payload["default_device"] = "cpu"
    application_paths.config_file.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    restored = manager.load()
    manager.save(restored)

    assert restored.device_preference == "cpu"
    assert manager.load().device_preference == "cpu"
    persisted = json.loads(application_paths.config_file.read_text(encoding="utf-8"))
    assert persisted["device_preference"] == "cpu"
    assert "resolved_device" not in persisted
