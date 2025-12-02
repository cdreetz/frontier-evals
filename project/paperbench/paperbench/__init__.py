"""
PaperBench - A benchmark for evaluating AI agents on scientific paper reproduction.

This module provides tools for running paper reproduction evaluations,
including task definitions, prompts, and setup utilities.

Quick Start (lightweight, minimal dependencies):
    from paperbench.api import get_initial_prompt, list_paper_ids, get_paper_info

    # Get the initial prompt for a rollout
    prompt = get_initial_prompt()

    # List all available papers
    papers = list_paper_ids()

    # Get info about a specific paper
    paper = get_paper_info("rice")

Full API (requires all dependencies):
    from paperbench import PaperBench, PBTask, paper_registry
"""

from __future__ import annotations

# Re-export the lightweight API functions (no heavy dependencies)
from paperbench.api import (
    # Types
    ChatMessage,
    PaperInfo,
    ChatCompletionToolParam,
    FunctionDefinition,
    ToolDefinition,
    RubricNode,
    PaperBenchExample,
    # Constants
    WORKSPACE_BASE,
    SUBMISSION_DIR,
    # Prompt functions
    get_instructions_path,
    get_initial_prompt,
    get_instructions_text,
    # Paper functions
    get_paper_info,
    list_paper_ids,
    get_papers_directory,
    get_paper_files,
    read_paper_markdown,
    read_paper_addendum,
    read_paper_rubric,
    # Setup function
    run_paper_setup,
    # Tool utilities - definitions
    get_oai_tools,
    get_tool_definitions,
    get_tool_names,
    is_submit_tool_call,
    # Tool utilities - execution
    execute_bash,
    execute_python,
    read_file_chunk,
    search_file,
    execute_submit,
    execute_tool,
    get_tool_map,
    # Rubric utilities
    get_rubric,
    get_rubric_dict,
    # Dataset utilities - legacy
    get_paperbench_dataset,
    get_paper_as_dict,
    # Dataset utilities - HuggingFace
    get_hf_dataset,
    get_paper_hf_row,
)

# Lazy imports for heavy dependencies
# These are only loaded when accessed, not at import time
def __getattr__(name: str):
    """Lazy import for heavy modules to avoid loading all dependencies at import time."""

    # Core evaluation classes
    if name == "PaperBench":
        from paperbench.nano.eval import PaperBench
        return PaperBench
    elif name == "PBTask":
        from paperbench.nano.task import PBTask
        return PBTask
    elif name == "PaperBenchGrade":
        from paperbench.nano.structs import PaperBenchGrade
        return PaperBenchGrade
    elif name == "PaperBenchResult":
        from paperbench.nano.structs import PaperBenchResult
        return PaperBenchResult
    elif name == "JudgeConfig":
        from paperbench.nano.structs import JudgeConfig
        return JudgeConfig
    elif name == "ReproductionConfig":
        from paperbench.nano.structs import ReproductionConfig
        return ReproductionConfig
    elif name == "ReproductionMetadata":
        from paperbench.nano.structs import ReproductionMetadata
        return ReproductionMetadata

    # Paper registry (uses pydantic, structlog)
    elif name == "Paper":
        from paperbench.paper_registry import Paper
        return Paper
    elif name == "PaperRegistry":
        from paperbench.paper_registry import PaperRegistry
        return PaperRegistry
    elif name == "paper_registry":
        from paperbench.paper_registry import paper_registry
        return paper_registry

    # Additional constants (duplicated in api.py but also available here)
    elif name == "LOGS_DIR":
        from paperbench.constants import LOGS_DIR
        return LOGS_DIR
    elif name == "AGENT_DIR":
        from paperbench.constants import AGENT_DIR
        return AGENT_DIR

    # Utility functions
    elif name == "get_root":
        from paperbench.utils import get_root
        return get_root
    elif name == "get_paperbench_data_dir":
        from paperbench.utils import get_paperbench_data_dir
        return get_paperbench_data_dir
    elif name == "get_experiments_dir":
        from paperbench.utils import get_experiments_dir
        return get_experiments_dir
    elif name == "get_agents_dir":
        from paperbench.utils import get_agents_dir
        return get_agents_dir
    elif name == "create_run_id":
        from paperbench.utils import create_run_id
        return create_run_id
    elif name == "create_run_dir":
        from paperbench.utils import create_run_dir
        return create_run_dir
    elif name == "get_default_runs_dir":
        from paperbench.utils import get_default_runs_dir
        return get_default_runs_dir
    elif name == "get_timestamp":
        from paperbench.utils import get_timestamp
        return get_timestamp

    raise AttributeError(f"module 'paperbench' has no attribute {name!r}")


__all__ = [
    # Lightweight API - Types
    "ChatMessage",
    "PaperInfo",
    "ChatCompletionToolParam",
    "FunctionDefinition",
    "ToolDefinition",
    "RubricNode",
    "PaperBenchExample",
    # Lightweight API - Constants
    "WORKSPACE_BASE",
    "SUBMISSION_DIR",
    # Lightweight API - Prompt functions
    "get_instructions_path",
    "get_initial_prompt",
    "get_instructions_text",
    # Lightweight API - Paper functions
    "get_paper_info",
    "list_paper_ids",
    "get_papers_directory",
    "get_paper_files",
    "read_paper_markdown",
    "read_paper_addendum",
    "read_paper_rubric",
    # Lightweight API - Setup function
    "run_paper_setup",
    # Lightweight API - Tool utilities (definitions)
    "get_oai_tools",
    "get_tool_definitions",
    "get_tool_names",
    "is_submit_tool_call",
    # Lightweight API - Tool utilities (execution)
    "execute_bash",
    "execute_python",
    "read_file_chunk",
    "search_file",
    "execute_submit",
    "execute_tool",
    "get_tool_map",
    # Lightweight API - Rubric utilities
    "get_rubric",
    "get_rubric_dict",
    # Lightweight API - Dataset utilities (legacy)
    "get_paperbench_dataset",
    "get_paper_as_dict",
    # Lightweight API - Dataset utilities (HuggingFace)
    "get_hf_dataset",
    "get_paper_hf_row",
    # Heavy classes (lazy loaded)
    "PaperBench",
    "PBTask",
    "PaperBenchGrade",
    "PaperBenchResult",
    "JudgeConfig",
    "ReproductionConfig",
    "ReproductionMetadata",
    "Paper",
    "PaperRegistry",
    "paper_registry",
    "LOGS_DIR",
    "AGENT_DIR",
    "get_root",
    "get_paperbench_data_dir",
    "get_experiments_dir",
    "get_agents_dir",
    "create_run_id",
    "create_run_dir",
    "get_default_runs_dir",
    "get_timestamp",
]
