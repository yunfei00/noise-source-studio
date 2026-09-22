"""Centralized, platform-aware application paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

APPLICATION_SLUG = "NoiseSourceStudio"
APPLICATION_AUTHOR = "NoiseSourceStudio"


@dataclass(frozen=True, slots=True)
class ApplicationPaths:
    """Filesystem locations used by the application."""

    data_directory: Path
    config_directory: Path
    log_directory: Path
    model_directory: Path
    output_directory: Path
    resource_directory: Path

    @classmethod
    def from_platformdirs(cls) -> ApplicationPaths:
        """Build paths in the current user's platform-specific application directories."""
        directories = PlatformDirs(APPLICATION_SLUG, APPLICATION_AUTHOR, roaming=False)
        data_directory = Path(directories.user_data_dir)
        return cls(
            data_directory=data_directory,
            config_directory=Path(directories.user_config_dir),
            log_directory=Path(directories.user_log_dir),
            model_directory=data_directory / "models",
            output_directory=data_directory / "outputs",
            resource_directory=Path(__file__).resolve().parents[3] / "resources",
        )

    @property
    def config_file(self) -> Path:
        """Return the persisted JSON configuration file."""
        return self.config_directory / "settings.json"

    @property
    def log_file(self) -> Path:
        """Return the main application log file."""
        return self.log_directory / "noise-source-studio.log"

    @property
    def history_directory(self) -> Path:
        """Return the user-local directory containing the history index and backups."""
        return self.data_directory / "history"

    @property
    def history_database(self) -> Path:
        """Return the SQLite history-index location."""
        return self.history_directory / "history.db"

    def ensure_directories(self) -> None:
        """Create writable application directories when they do not yet exist."""
        for directory in (
            self.data_directory,
            self.config_directory,
            self.log_directory,
            self.model_directory,
            self.output_directory,
            self.history_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
