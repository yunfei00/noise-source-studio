"""Application page widgets."""

from noise_source_studio.presentation.pages.batch_prediction_page import (
    BatchPredictionPage,
)
from noise_source_studio.presentation.pages.dashboard_page import DashboardPage
from noise_source_studio.presentation.pages.history_page import HistoryPage
from noise_source_studio.presentation.pages.log_page import LogPage
from noise_source_studio.presentation.pages.model_management_page import (
    ModelManagementPage,
)
from noise_source_studio.presentation.pages.settings_page import SettingsPage
from noise_source_studio.presentation.pages.single_prediction_page import (
    SinglePredictionPage,
)
from noise_source_studio.presentation.pages.validation_page import ValidationPage

__all__ = [
    "BatchPredictionPage",
    "DashboardPage",
    "HistoryPage",
    "LogPage",
    "ModelManagementPage",
    "SettingsPage",
    "SinglePredictionPage",
    "ValidationPage",
]
