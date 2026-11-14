"""cfisac -- AP deployment for localizing accuracy and user rate in cell-free ISAC systems.

Official implementation of Xu et al., ACM MobiCom '24 (doi:10.1145/3636534.3698221).
"""

from cfisac.env import APDeploymentEnv
from cfisac.metrics import (
    ObjectiveBreakdown,
    evaluate_deployment,
    fim_determinant,
    snr,
    user_rates,
)
from cfisac.scenario import ScenarioConfig, archived_scenario

__version__ = "1.0.0"

__all__ = [
    "APDeploymentEnv",
    "ObjectiveBreakdown",
    "ScenarioConfig",
    "archived_scenario",
    "evaluate_deployment",
    "fim_determinant",
    "snr",
    "user_rates",
    "__version__",
]
