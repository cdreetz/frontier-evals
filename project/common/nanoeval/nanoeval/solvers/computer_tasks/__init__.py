"""
Computer tasks module for nanoeval.

Provides the core interfaces for computer-based evaluation tasks,
including ComputerInterface, ComputerTask, and related classes.
"""

from __future__ import annotations

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    JupyterComputerInterface,
    ComputerConfiguration,
    ComputerRuntime,
    RuntimeConfig,
    NetworkMode,
    ExecutionResult,
    JupyterExecutionResult,
    ContainerResources,
    VolumeMount,
)

from nanoeval.solvers.computer_tasks.task import (
    ComputerTask,
    SimpleJupyterTask,
    Grade,
)

from nanoeval.solvers.computer_tasks.solver import (
    PythonCodingEval,
    PythonCodingSolver,
)

from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    Step,
)

__all__ = [
    # Interfaces
    "ComputerInterface",
    "JupyterComputerInterface",
    "ComputerConfiguration",
    "ComputerRuntime",
    "RuntimeConfig",
    "NetworkMode",
    "ExecutionResult",
    "JupyterExecutionResult",
    "ContainerResources",
    "VolumeMount",
    # Tasks
    "ComputerTask",
    "SimpleJupyterTask",
    "Grade",
    # Solver
    "PythonCodingEval",
    "PythonCodingSolver",
    # Steps
    "FinalResult",
    "Step",
]
