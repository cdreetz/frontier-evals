"""
Nanoeval Alcatraz integration module.

Provides the AlcatrazComputerRuntime for running computer tasks
in Alcatraz containers.
"""

from __future__ import annotations

from nanoeval_alcatraz.alcatraz_computer_interface import (
    AlcatrazComputerRuntime,
    AlcatrazComputerInterface,
)

from nanoeval_alcatraz.task_to_alcatraz_config import (
    task_to_alcatraz_config,
)

__all__ = [
    "AlcatrazComputerRuntime",
    "AlcatrazComputerInterface",
    "task_to_alcatraz_config",
]
