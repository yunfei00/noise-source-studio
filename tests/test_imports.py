"""Smoke-test public modules without external data or model dependencies."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "noise_source_studio.application",
        "noise_source_studio.common.paths",
        "noise_source_studio.domain.interfaces.inference_engine",
        "noise_source_studio.infrastructure.config.settings",
        "noise_source_studio.infrastructure.logging.setup",
        "noise_source_studio.presentation.main_window",
        "noise_source_studio.presentation.navigation",
        "noise_source_studio.presentation.pages",
    ),
)
def test_main_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name)
