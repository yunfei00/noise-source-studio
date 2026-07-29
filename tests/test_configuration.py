"""Configuration and platform path tests."""

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
    settings = manager.load().model_copy(update={"default_device": "cpu"})

    manager.save(settings)

    assert manager.load().default_device == "cpu"
