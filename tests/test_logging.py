"""Logging setup tests."""

import logging

from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.infrastructure.logging import configure_logging


def test_log_file_is_created(application_paths: ApplicationPaths) -> None:
    log_file = configure_logging(application_paths.log_directory, "INFO")
    logger = logging.getLogger("noise_source_studio")

    logger.info("test startup message")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.is_file()
    assert "test startup message" in log_file.read_text(encoding="utf-8")
